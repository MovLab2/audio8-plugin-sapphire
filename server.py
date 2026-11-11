"""Audio8 TTS server — zero-shot voice cloning in its own conda env.

Runs as a Sapphire plugin service (capabilities.services) with the
sapphire-plugin-audio8 env's python on SAPPHIRE_SERVICE_PORT.
Model: Edge0/Audio8-TTS-Realistic-Air (601M DualAR, 44.1kHz, 23 languages,
Apache-2.0; finetune of Audio8-TTS-Preview-0.6b, same remote code),
pinned to a commit sha so future repo pushes can't run new remote code here.

Voices live in user/audio8-voices/: <name>.wav (5-15s clean speech) beside
<name>.txt holding the exact transcript — the model hard-requires the
transcript for cloning. Built-ins ship in this plugin's voices/ folder; the
user's folder wins on a name clash. No resolvable voice -> the model's base voice.

Speed: the model has no rate knob but copies the reference clip's pace, so
the reference is time-stretched (rubberband via pedalboard) and the model
speaks at that pace natively; post-stretching the output sounded processed
(A/B 2026-11-08). Pitch is varispeed like Kokoro/Piper, folded into the
Opus resample. The base voice (no reference) has no speed control.

Engine: fastgen.py replays each decode frame as one CUDA graph, fused by
torch.compile when Triton works. Picked once at load, best rung first:
tuned (compile with kernels autotuned for this GPU) -> compiled -> graph ->
original model.generate; a rung that fails to warm up
drops to the next, and a runtime failure demotes to original for good.

Shared GPU: run alone on another computer (audio8-server.sh / .bat), it
also opens a TLS door (worker_door.py, a copy of Sapphire's) and prints a
pairing string for the computer named "sapphire", the same on every start.
A Sapphire in Client only mode sends each voice by the hash of its clip and
transcript; an unknown hash gets 409 and the clip is sent once (POST /voice),
kept in user/audio8-received/ (newest RECEIVED_MAX).
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import hashlib
import io
import json
import logging
import os
import re
import select
import ssl
import sys
import threading
import time

import worker_door      # a copy of Sapphire's core/worker_door.py: this server also runs alone

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("audio8")

# Download over HF's plain CDN, not xet: xet's CAS pool served bad TLS certs
# (2026-11-08) and stalled the first load forever. Set before the lazy
# transformers/huggingface_hub import in _load_model reads it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_ID = "Edge0/Audio8-TTS-Realistic-Air"
MODEL_REV = "23c305cc013fde378edb459f538317043c993a74"
PORT = int(os.environ.get("SAPPHIRE_SERVICE_PORT", 5014))
ROOT = os.environ.get("SAPPHIRE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
VOICES_DIR = os.path.join(ROOT, "user", "audio8-voices")
BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
# Voices other computers sent, by hash, and the door's cert and keys.
RECEIVED_DIR = os.path.join(ROOT, "user", "audio8-received")
RECEIVED_MAX = 64
DOOR = os.path.join(ROOT, "user", "worker_door", "audio8")
HASH_RE = re.compile(r"^h-[0-9a-f]{64}$")
MAX_VOICE_BYTES = 8 * 1024 * 1024        # a 30 s clip at 44.1 kHz is ~2.6 MB, base64 ~3.5
_START = json.loads(os.environ.get("SAPPHIRE_PLUGIN_SETTINGS") or "{}")
# torch.compile's kernel caches: rebuildable, so under user/models/ (never
# backed up). Cold compile ~40s, warm ~15s; without a cache, every start pays 40.
_CACHE = os.path.join(ROOT, "user", "models", "audio8")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE, "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE, "triton"))
# Grow and give back VRAM in pages instead of caching every odd-sized block
# a decode ever asked for (reserved ran 7 GB for a 4.4 GB peak).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
VOICE_RE = re.compile(r"^[A-Za-z0-9_-]+$")  # blocks path traversal via voice name
MAX_GROUP_CHARS = 300
SPEED_MIN, SPEED_MAX = 0.75, 1.6   # the model follows a stretched reference well in this band
PITCH_MIN, PITCH_MAX = 0.5, 2.0
# A request during the load waits this long, then gets 503 with the phase:
# a warm start (~15 s) still answers, a first-ever download says why it can't.
READY_TIMEOUT = 60
# What the model reads at MODEL_REV (the repo's badges and banner are skipped).
# codec.pth is 1.3 GB that the model would otherwise fetch lazily on the first
# request, AFTER saying ready (Prime, 2026-11-09).
FILES = ("config.json", "generation_config.json", "configuration_arktts.py",
         "modeling_arktts.py", "modeling_arktts_codec.py", "processing_arktts.py",
         "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "model.safetensors", "codec.pth")

# /health. Core reads status/phase/progress/error (core/service_health.py).
_state = {"status": "loading", "phase": "Starting", "progress": None, "error": None, "engine": None}
_ready = threading.Event()
_gpu_lock = threading.Lock()
_model = None
_processor = None
_device = "cpu"
_fast = None               # FastGen, or None on the original engine
_ref_cache = {}            # (wav, mtime, speed) -> {"array", "sampling_rate"}
_ref_lock = threading.Lock()
_REF_CACHE_MAX = 32
_code_cache = {}           # (wav, mtime, speed) -> the reference's codec codes, encoded once
_lead = None               # codec codes of silence, decoded ahead of each group


def _fetch():
    """Download whatever of FILES isn't cached, its bytes counted into /health
    by watching the hub's .incomplete blobs grow (any hub version; 0.36's
    hf_hub_download takes no progress hook). All cached = no network at all."""
    import glob
    from huggingface_hub import HfApi, constants, hf_hub_download, try_to_load_from_cache
    missing = [f for f in FILES
               if not isinstance(try_to_load_from_cache(MODEL_ID, f, revision=MODEL_REV), str)]
    if not missing:
        return
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REV, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    total = sum(sizes.get(f, 0) for f in missing)
    blobs = os.path.join(constants.HF_HUB_CACHE, "models--" + MODEL_ID.replace("/", "--"), "blobs")
    state = {"base": 0, "on": True}

    def count():
        while state["on"]:
            partial = 0
            for path in glob.glob(os.path.join(blobs, "*.incomplete")):
                try:
                    partial += os.path.getsize(path)
                except OSError:
                    pass                          # finished between the glob and the stat
            if big:
                _state["progress"] = {"done": min(total, state["base"] + partial), "total": total}
            threading.Event().wait(0.5)

    big = total > 50e6
    _state.update(phase=f"Downloading the model ({total / 1e9:.1f} GB, first start only)" if big
                  else "Fetching model files", progress={"done": 0, "total": total} if big else None)
    logger.info(f"downloading {len(missing)} file(s), {total / 1e9:.2f} GB")
    counter = threading.Thread(target=count, daemon=True)
    counter.start()
    try:
        for f in missing:
            hf_hub_download(MODEL_ID, f, revision=MODEL_REV)
            state["base"] += sizes.get(f, 0)
    finally:
        state["on"] = False
        counter.join(2)
        _state["progress"] = None


def _load_model():
    global _model, _processor, _device
    try:
        import torch
        try:
            _fetch()
        except Exception as e:      # offline with a usable cache: from_pretrained says if it isn't
            logger.warning(f"model download check failed, loading from cache: {e!r}")
            _state["progress"] = None
        from transformers import AutoModel, AutoProcessor
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if _device == "cuda" else torch.float32
        _state["phase"] = "Loading the model onto the GPU" if _device == "cuda" else "Loading the model (CPU)"
        logger.info(f"Loading {MODEL_ID}@{MODEL_REV[:8]} on {_device}")
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REV, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REV, trust_remote_code=True, dtype=dtype
        ).eval().to(_device)
        _state["phase"] = "Warming up the engine (first start tunes for this GPU, a few minutes)"
        _warm_engine()
        # One real line end to end: reference encode, decode, codec. Ready
        # then means the next request does no first-time work.
        _state["phase"] = "Warming up the voice"
        try:
            warm = _render("Ready.", "sapphire", 0.8)
        except ValueError:                       # a user 'sapphire' without its transcript
            warm = _render("Ready.", None, 0.8)
        for _ in warm:
            pass
        _state.update(status="ready", phase="Ready")
        logger.info(f"Model ready (engine: {_state['engine']})")
        if _banner:
            print(_banner, flush=True)           # again: the load's logs scrolled the first one away
    except Exception as e:
        _state.update(status="error", phase="Failed", progress=None, error=str(e) or type(e).__name__)
        logger.error(f"Model load failed: {e!r}")
    finally:
        _ready.set()


def _warm_engine():
    """Pick the fastest decode engine that warms up cleanly. Runs inside the
    load, before ready: warm-up shares the model's KV caches with every
    request, so nothing could run beside it anyway."""
    global _fast
    import time
    import torch
    from fastgen import FastGen
    if _device != "cuda":
        _state["engine"] = "original"
        return
    warm = _processor(text=["Warming up."], return_tensors="pt")
    warm = {k: v.to(_device) for k, v in warm.items()}
    pristine = _model._setup_generation_caches
    # "tuned" benchmarks Triton kernels on this GPU for the frame's matmuls:
    # ~38% less GPU time per frame than cuBLAS's bf16 gemv (2026-11-10, 5090),
    # tuned once and cached under _CACHE. Its log shows "out of resource"
    # lines for kernel configs too big for this card: those are skipped.
    modes = {"tuned": "max-autotune-no-cudagraphs", "compiled": None}
    for name in ("tuned", "compiled", "graph"):
        t = time.time()
        try:
            fg = FastGen(_model)
            if name in modes:
                kw = {"mode": modes[name]} if modes[name] else {}
                fg._frame = torch.compile(fg._frame, dynamic=False, **kw)
            fg.generate(warm, max_new_tokens=8)        # compile + capture
            fg.generate(warm, max_new_tokens=8)        # and a real replay
            _fast = fg
            _state["engine"] = name
            logger.info(f"engine {name} ready in {time.time() - t:.1f}s")
            return
        except Exception as e:
            logger.warning(f"engine {name} unavailable ({time.time() - t:.1f}s): {e!r}")
            _model._setup_generation_caches = pristine
            torch.cuda.empty_cache()
    _state["engine"] = "original"


def _demote(err):
    """A fast-path failure mid-request: drop to the original engine for the
    life of this process, loudly. The model's own cache setup comes back."""
    global _fast
    logger.error(f"engine {_state['engine']} failed, using original from now on: {err!r}")
    _fast = None
    if "_setup_generation_caches" in _model.__dict__:
        del _model._setup_generation_caches
    _state["engine"] = "original"


def _codes(inputs, temperature):
    """Decode codes [1, codebooks, T] for one group on the current engine."""
    if _fast is not None:
        try:
            return _fast.generate(inputs, max_new_tokens=1024, temperature=temperature, top_p=0.95)
        except Exception as e:
            _demote(e)
    out = _model.generate(**inputs, max_new_tokens=1024, temperature=temperature,
                          top_p=0.95, top_k=50, do_sample=True, return_dict_in_generate=True)
    return out.codes


# Each group is decoded after LEAD frames of encoded silence, then those are
# cut: a codec started cold wakes up mid-word (a hard click of an onset at
# every group); seeded with silence it starts the way speech after a quiet
# room does.
LEAD = 8
# The codec's conv decoder runs in pieces of DECODE_CHUNK latent steps (512
# samples each, ~3 s) with DECODE_CTX steps of lead-in: in strict fp32, 64
# matches a whole decode exactly (16 is 96 dB, 4 is audible).
DECODE_CHUNK, DECODE_CTX = 256, 64


def _decode(codes, start=0):
    """Codes [1, codebooks, T] -> float32 audio (on the codec's device), from
    frame `start` on: the codec is causal, so the audio of frames before
    `start` is skipped without changing what follows.

    The memory of a decode is the conv decoder working at audio rate: about
    1 GB per 20 s of speech, 4.4 GB peak for a full 1024-frame group. Its
    transformer runs whole (it looks back ~1000 frames: chunking it is
    audible); the convs run in ~3 s pieces, so peak VRAM stays flat at any
    length, and the pieces run a little faster than one big pass. Same
    samples as a whole decode, to float rounding (2026-11-10)."""
    global _lead
    import torch
    codec = _model.load_codec(device=codes.device)
    item = codes[0]
    n = int((item >= 0).all(dim=0).sum().item())
    if n == 0:
        return torch.empty(0, device=codes.device)
    with torch.inference_mode():
        if _lead is None or _lead.device != codes.device:
            quiet = torch.zeros((1, 1, codec.frame_length * 40), device=codes.device,
                                dtype=next(codec.parameters()).dtype)
            _lead = codec.encode(quiet)[0][0, :, -LEAD:].long()
        latent = codec.quantizer.decode(torch.cat((_lead, item[:, :n].long()), dim=-1).unsqueeze(0))
        hop = codec.hop_length
        per = codec.frame_length // hop               # latent steps per frame
        out = []
        for s in range((LEAD + start) * per, latent.shape[-1], DECODE_CHUNK):
            a = max(0, s - DECODE_CTX)
            wav = codec.decoder(latent[..., a:s + DECODE_CHUNK])[0, 0].float()
            out.append(wav[(s - a) * hop:])
        return torch.cat(out) if out else torch.empty(0, device=codes.device)


# /tts/stream sends each group as several OGGs while it decodes, each cut in
# a pause the model left: the browser starts every segment with a ~20-50 ms
# gap, heard only mid-word. A segment runs SEG_MIN s at least; with no pause
# by SEG_MAX s it ends at the quietest 20 ms past SEG_MIN.
STEP = 12                     # frames between incremental decodes (~0.56 s)
SEG_MIN, SEG_MAX, PAUSE_S = 1.5, 6.0, 0.16
# Frames are paced to PACE_RATE x realtime so the GPU is shared smoothly
# instead of in one burst (2026-11-10, a probe needing 20 of every 33 ms: a
# burst made it miss 85 of 86 frames, paced 2x none), but only with PACE_LEAD
# s of sent audio queued ahead of playback: a segment waits for a pause, up to
# SEG_MAX s, and must still land before the queue runs dry (0.8.0 paced from
# 1.5 s and stuttered on a 5060 Ti beside an avatar). A GPU that can't beat
# PACE_RATE comfortably is never paced.
PACE_RATE, PACE_LEAD = 1.5, 6.0
FRAME_S = 2048 / 44100


class _Pacer:
    """One stream's clock (playback taken to start at its first segment) and
    this GPU's measured speed, sleeps left out (`speed`, x realtime)."""

    def __init__(self, on):
        self.on, self.t0, self.sent, self.tick = on, None, 0.0, time.monotonic()
        self.per = None               # seconds of compute per frame, smoothed

    @property
    def speed(self):
        return FRAME_S / self.per if self.per else 1.0

    def first_min(self):
        """Seconds the first segment must hold: enough to cover making the
        next one (up to SEG_MIN + SEG_MAX s) at this GPU's speed."""
        return min(8.0, max(SEG_MIN, (SEG_MIN + SEG_MAX) / self.speed + 0.5))

    def sent_audio(self, seconds):
        if self.t0 is None:
            self.t0 = time.monotonic()
        self.sent += seconds

    def frame(self):
        now = time.monotonic()
        dt = now - self.tick
        self.per = dt if self.per is None else 0.9 * self.per + 0.1 * dt
        if (self.on and self.t0 is not None and self.speed > PACE_RATE * 1.3
                and self.sent - (now - self.t0) > PACE_LEAD):
            wait = FRAME_S / PACE_RATE - dt
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
        self.tick = now


def _cut(audio, sr, min_s=SEG_MIN):
    """Sample index to end a segment at, or None to wait for more: the middle
    of the last pause past min_s (PAUSE_S of 20 ms windows under 5% of the
    loudest), or past min_s + SEG_MAX the quietest window past min_s."""
    import numpy as np
    win = sr // 50
    n = len(audio) // win
    lo = int(min_s * 50)
    if n <= lo:
        return None
    rms = np.sqrt((audio[:n * win].reshape(n, win) ** 2).mean(axis=1))
    quiet = rms < max(0.003, 0.05 * rms.max())
    best, run = None, 0
    for i in range(n + 1):
        if i < n and quiet[i]:
            run += 1
            continue
        mid = i - run + run // 2
        if run >= PAUSE_S * 50 and mid >= lo:
            best = mid
        run = 0
    if best is None and n >= lo + SEG_MAX * 50:
        best = lo + int(np.argmin(rms[lo:]))
    return None if best is None else best * win + win // 2


def _segments(inputs, temperature, pacer):
    """One group's audio in segments cut at pauses while it decodes. Each
    step decodes the codes so far from the last decoded frame on: the same
    samples as one whole decode (fp32, 2026-11-10)."""
    import numpy as np
    import torch
    sr = _model.config.codec_sample_rate
    frames, done, pending = [], 0, np.zeros(0, dtype="float32")
    it = _fast.stream(inputs, max_new_tokens=1024, temperature=temperature, top_p=0.95)
    while True:
        try:
            f = next(it)
        except StopIteration:
            break
        except Exception as e:
            _demote(e)
            if frames:
                raise
            yield _decode(_codes(inputs, temperature)).cpu().numpy()
            return
        frames.append(f)
        pacer.frame()
        if len(frames) - done >= STEP:
            new = _decode(torch.stack(frames, dim=2), start=done).cpu().numpy()
            pending, done = np.concatenate([pending, new]), len(frames)
            cut = _cut(pending, sr, SEG_MIN if pacer.t0 is not None else pacer.first_min())
            if cut:
                yield pending[:cut]
                pending = pending[cut:]
    if len(frames) > done:
        new = _decode(torch.stack(frames, dim=2), start=done).cpu().numpy()
        pending = np.concatenate([pending, new])
    if len(pending):
        yield pending


def _ref_codes(inputs, key):
    """The processor's reference audio swapped for its codec codes, encoded
    once per (clip, mtime, speed) instead of on every group of every request."""
    with _ref_lock:
        hit = _code_cache.get(key)
    if hit is None:
        hit = _model.encode_audio(inputs["reference_audio_values"], inputs["reference_audio_lengths"])
        with _ref_lock:
            if len(_code_cache) >= _REF_CACHE_MAX:
                _code_cache.clear()
            _code_cache[key] = hit
    inputs = {k: v for k, v in inputs.items() if not k.startswith("reference_audio_")}
    inputs["reference_codes"], inputs["reference_code_lengths"] = hit
    return inputs


_banner = ""            # the pairing box, printed at the start and again when ready


def _private(text):
    """Words as another computer sees them: without this computer's folders or user name."""
    text = str(text).replace(ROOT, "<audio8>")
    home = os.path.expanduser("~")
    return text.replace(home, "~") if len(home) > 1 else text


class NeedVoice(LookupError):
    """A hashed voice this computer hasn't been sent yet: answered 409."""


def voice_hash(wav_bytes, transcript):
    """The name a voice goes by between computers (provider computes the same)."""
    return "h-" + hashlib.sha256(wav_bytes + b"\0" + transcript.strip().encode("utf-8")).hexdigest()


def _resolve_voice(name):
    """Return (wav_path, transcript) or (None, None) for base voice."""
    if not name or not VOICE_RE.match(name):
        return None, None
    if HASH_RE.match(name):
        if not all(os.path.isfile(os.path.join(RECEIVED_DIR, f"{name}{x}")) for x in (".wav", ".txt")):
            raise NeedVoice(name)               # never sent, pruned, or mid-prune: sent again
        dirs = (RECEIVED_DIR,)
    else:
        dirs = (VOICES_DIR, BUILTIN_DIR)
    folder = next((d for d in dirs if os.path.isfile(os.path.join(d, f"{name}.wav"))), None)
    if folder is None:
        return None, None
    wav = os.path.join(folder, f"{name}.wav")
    txt = os.path.join(folder, f"{name}.txt")
    if not os.path.isfile(txt):
        raise ValueError(f"voice '{name}' has no transcript — add {name}.txt "
                         "with the exact words spoken in the wav")
    with open(txt, encoding="utf-8", errors="replace") as fh:
        transcript = fh.read().strip()
    if not transcript:
        raise ValueError(f"voice '{name}': {name}.txt is empty")
    return wav, transcript


def _clamp(value, lo, hi, default=1.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if v != v else max(lo, min(hi, v))   # NaN -> default


def _reference(wav_path, speed):
    """The reference clip at `speed` (1.0 = the file as-is, as a path). The
    stretch is cached per (file, mtime, speed); speed is rounded to 0.05 so
    a slider drag doesn't fill the cache."""
    speed = round(speed * 20) / 20
    if speed == 1.0:
        return wav_path
    key = (wav_path, os.path.getmtime(wav_path), speed)
    with _ref_lock:
        hit = _ref_cache.get(key)
    if hit is not None:
        return hit
    try:
        import pedalboard
    except ImportError:
        logger.warning("pedalboard missing (rebuild the Audio8 environment) — speaking at the clip's own pace")
        return wav_path
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    stretched = pedalboard.time_stretch(mono[None, :], sr, stretch_factor=speed)[0]
    ref = {"array": stretched.astype("float32"), "sampling_rate": sr}
    with _ref_lock:
        if len(_ref_cache) >= _REF_CACHE_MAX:
            _ref_cache.clear()
        _ref_cache[key] = ref
    logger.info(f"reference {os.path.basename(wav_path)} stretched to {speed}x")
    return ref


def _split_text(text):
    """Split into sentence groups of <= MAX_GROUP_CHARS so each generate
    stays well inside the model's token budget. An over-long single
    sentence passes through whole."""
    sents = re.split(r"(?<=[.!?…])\s+", text.strip())
    groups, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > MAX_GROUP_CHARS:
            groups.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        groups.append(cur)
    return groups or [text]


def _render(text, voice, temperature, speed=1.0, pitch=1.0, stream=False, wanted=None):
    """Yield float32 audio per sentence group (~MAX_GROUP_CHARS), at the codec
    rate, each group after the first led by a short breath gap. stream=True
    (on the fast engine) yields each group in paced segments as it decodes
    (_segments, _Pacer). Voice and
    reference problems raise ValueError BEFORE the first yield, so a caller
    can still answer 400. The GPU lock spans the whole request: one text's
    groups never interleave with another request's. `wanted()` False = the
    caller left (a Stop): no group starts for nobody, nor the first one after
    a wait for the lock."""
    import numpy as np
    import torch

    wav_path, ref_text = _resolve_voice(voice)
    ref_audio = _reference(wav_path, speed) if wav_path else None
    ref_key = (wav_path, os.path.getmtime(wav_path), round(speed * 20) / 20) if wav_path else None
    sr = _model.config.codec_sample_rate
    gap = np.zeros(int(0.12 * sr), dtype="float32")

    def groups():
        with _gpu_lock:
            for n, chunk in enumerate(_split_text(text)):
                if wanted and not wanted():
                    logger.info("the caller left: the rest of its text is not made")
                    return
                kwargs = {"text": [chunk], "return_tensors": "pt"}
                if wav_path:
                    kwargs["reference_audio"] = [ref_audio]
                    kwargs["reference_text"] = [ref_text]
                inputs = _processor(**kwargs)
                inputs = {k: v.to(_device) for k, v in inputs.items()}
                if ref_key and "reference_audio_values" in inputs:
                    inputs = _ref_codes(inputs, ref_key)
                with torch.inference_mode():
                    if stream and _fast is not None:
                        parts = _segments(inputs, temperature, pacer)
                    else:
                        parts = iter([_decode(_codes(inputs, temperature)).cpu().numpy()])
                    for i, audio in enumerate(parts):
                        if n and not i:
                            audio = np.concatenate([gap, audio])
                        pacer.sent_audio(len(audio) / sr)
                        yield audio
    pacer = _Pacer(stream)
    return groups()


def _encode(audio, pitch=1.0, fmt="ogg"):
    """Codec-rate float32 -> one OGG/Opus file (48 kHz), or a 16-bit WAV at
    the codec rate (lossless, for downloads: MP3 made from Opus would be
    lossy twice). Varispeed pitch folded into the resample: reading the
    samples as if recorded at sr*pitch makes pitch<1 deeper and longer.
    Source rate snapped to 100 Hz keeps the resample kernel small."""
    import soundfile as sf
    import torch
    sr = _model.config.codec_sample_rate
    rate = 48000 if fmt == "ogg" else sr
    src = int(round(sr * pitch / 100)) * 100
    if src != rate:
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), src, rate).numpy()
    buf = io.BytesIO()
    if fmt == "ogg":
        sf.write(buf, audio, rate, format="OGG", subtype="OPUS")
    else:
        sf.write(buf, audio.clip(-1, 1), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _synth(text, voice, temperature, speed=1.0, pitch=1.0, fmt="ogg"):
    """The whole text as one blob (fmt "ogg" or "wav")."""
    import numpy as np
    pieces = list(_render(text, voice, temperature, speed, pitch))
    full = np.concatenate(pieces) if pieces else np.zeros(1, dtype="float32")
    return _encode(full, pitch, fmt)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # chunked /tts/stream; every other reply sets Content-Length
    def log_message(self, fmt, *args):  # route http.server noise to our logger
        logger.debug(fmt % args)

    def _json(self, code, obj):
        if getattr(self.server, "door", None):
            obj = {k: _private(v) if isinstance(v, str) else v for k, v in obj.items()}
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _let_in(self):
        """Through the shared door a request needs one of the door's keys
        (refused and answered here); on 127.0.0.1 it's Sapphire's own call."""
        if getattr(self.server, "door", None) and not worker_door.admit(self):
            # Read the body first: closing with it unread resets the connection
            # under the reply, and the caller never sees why.
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            if 0 < length <= MAX_VOICE_BYTES * 2:
                self.rfile.read(length)
            self.close_connection = True
            self._json(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self._let_in():
            return
        if self.path == "/health":
            self._json(200, _state)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._let_in():
            return
        door = getattr(self.server, "door", None)
        if self.path not in ("/tts", "/tts/stream") and not (door and self.path == "/voice"):
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_VOICE_BYTES * 2:
                self.close_connection = True
                self._json(413, {"error": "too big"})
                return
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError
        except Exception:
            self._json(400, {"error": "bad json"})
            return
        if self.path == "/voice":
            self._receive(data)
            return
        text = (data.get("text") or "").strip()
        if not text:
            self._json(400, {"error": "empty text"})
            return
        if not _ready.wait(timeout=READY_TIMEOUT) or _state["status"] != "ready":
            why = _state["error"] if _state["status"] == "error" else _state["phase"]
            self._json(503, {"error": f"Audio8 isn't ready: {why}"})
            return
        temperature = _clamp(data.get("temperature") or 0.8, 0.05, 2.0, 0.8)
        speed = _clamp(data.get("speed", 1.0), SPEED_MIN, SPEED_MAX)
        pitch = _clamp(data.get("pitch", 1.0), PITCH_MIN, PITCH_MAX)
        if self.path == "/tts/stream":
            self._stream(text, data.get("voice"), temperature, speed, pitch,
                         pieces=data.get("pieces") is True)
            return
        try:
            fmt = "wav" if data.get("format") == "wav" else "ogg"
            audio = _synth(text, data.get("voice"), temperature, speed, pitch, fmt)
        except NeedVoice as e:
            self._json(409, {"error": "voice not here yet", "need": str(e)})
            return
        except ValueError as e:  # voice/transcript problems — user-fixable
            logger.error(str(e))
            self._json(400, {"error": str(e)})
            return
        except Exception as e:
            logger.error(f"synthesis failed: {e!r}")
            self._json(500, {"error": str(e)})
            return
        self.send_response(200)
        self.send_header("Content-Type", f"audio/{fmt}")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def _still_there(self):
        """The caller sends nothing after its request: a readable socket now is
        its close (a FIN, or TLS's close_notify). A write would only fail a
        group or two later, each one GPU seconds made for nobody."""
        try:
            return not select.select([self.connection], [], [], 0)[0]
        except (OSError, ValueError):
            return False

    def _stream(self, text, voice, temperature, speed, pitch, pieces=False):
        """One complete OGG per segment (a pause-cut piece of a group on the
        fast engine, else a whole group), written the moment it's made
        (Kokoro's /tts/stream shape; the provider splits files at EOS pages).
        Chunked transfer, one HTTP chunk per OGG: a close-delimited body made
        requests' iter_content wait for the END before yielding anything.
        A failure after the 200 can only end the body early."""
        try:
            groups = _render(text, voice, temperature, speed, pitch, stream=pieces, wanted=self._still_there)
        except NeedVoice as e:
            self._json(409, {"error": "voice not here yet", "need": str(e)})
            return
        except ValueError as e:
            logger.error(str(e))
            self._json(400, {"error": str(e)})
            return
        except Exception as e:
            logger.error(f"synthesis failed: {e!r}")
            self._json(500, {"error": str(e)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/ogg")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for audio in groups:
                ogg = _encode(audio, pitch)
                self.wfile.write(b"%x\r\n%s\r\n" % (len(ogg), ogg))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (ConnectionError, TimeoutError, ssl.SSLError):   # Windows: ConnectionAbortedError; TLS: SSLEOFError
            logger.info("stream client went away")       # a Stop: the pump closed it
        except Exception as e:
            logger.error(f"stream synthesis failed: {e!r}")
        finally:
            groups.close()                                 # releases the GPU lock now


    def _receive(self, data):
        """A voice from another computer: {wav: base64, text}. Kept under its
        hash, which is computed here, so a name can't point at another clip."""
        try:
            wav = base64.b64decode(data.get("wav") or "", validate=True)
        except Exception:
            wav = b""
        text = str(data.get("text") or "").strip()
        if not wav.startswith(b"RIFF") or len(wav) > MAX_VOICE_BYTES or not text:
            self._json(400, {"error": "a voice needs a WAV clip and its transcript"})
            return
        name = voice_hash(wav, text)
        try:
            os.makedirs(RECEIVED_DIR, exist_ok=True)
            for ext, body in ((".txt", text.encode("utf-8")), (".wav", wav)):
                final = os.path.join(RECEIVED_DIR, f"{name}{ext}")
                if os.path.isfile(final):        # same name = same bytes: never replace a file a synth may be reading (Windows refuses)
                    continue
                tmp = f"{final}.{os.getpid()}-{threading.get_ident()}.tmp"
                with open(tmp, "wb") as fh:
                    fh.write(body)
                os.replace(tmp, final)
        except OSError as e:
            logger.error(f"could not keep a received voice: {e!r}")
            self._json(500, {"error": f"the GPU computer could not save the voice: {e}"})
            return

        def age(f):
            try:
                return os.path.getmtime(os.path.join(RECEIVED_DIR, f))
            except OSError:
                return 0.0
        kept = sorted((f for f in os.listdir(RECEIVED_DIR) if f.endswith(".wav")), key=age, reverse=True)
        for old in kept[RECEIVED_MAX:]:
            for ext in (".wav", ".txt"):
                try:
                    os.remove(os.path.join(RECEIVED_DIR, old[:-4] + ext))
                except OSError:
                    pass
        logger.info(f"received a voice ({len(wav) // 1024} KB), kept as {name[:14]}")
        self._json(200, {"voice": name})


def _open_door():
    """Run alone (the launchers set share): other computers' Sapphires use this
    GPU through a TLS door, a key per computer. "sapphire" is let in on the
    first start and its pairing string printed on every start: the same
    string until its key is removed or the door's folder is deleted."""
    global _banner
    port = int(_START.get("share_port") or 5015)
    try:
        if "sapphire" not in worker_door.clients(DOOR):
            worker_door.add_client(DOOR, "sapphire")
        line = worker_door.pairing(DOOR, "sapphire", f"{worker_door.guess_host()}:{port}")
        srv = worker_door.serve(ThreadingHTTPServer, Handler, "0.0.0.0", port, DOOR)
    except Exception as e:
        logger.error(f"Shared door not opened on port {port}: {e}")
        print(f"\nAudio8 could NOT open port {port} for the other computer: {e}\n"
              "Is Audio8 already running in another window? Close this one.\n", flush=True)
        raise SystemExit(1)                      # run alone, the door is the whole point: never "ready" without it
    threading.Thread(target=srv.serve_forever, daemon=True, name="audio8-door").start()
    bar = "=" * 72
    _banner = (f"\n{bar}\nAudio8 is shared on port {port}. In Sapphire: Settings > Audio8,\n"
               f"turn on Client only and paste this pairing string:\n\n{line}\n\n"
               f"It stays the same on every start. Allow port {port} in the firewall.\n{bar}\n")
    print(_banner, flush=True)


def main():
    os.makedirs(VOICES_DIR, exist_ok=True)
    if _START.get("share"):
        _open_door()                             # first: a second copy stops here, before loading a model

    class Local(ThreadingHTTPServer):
        allow_reuse_address = sys.platform != "win32"     # Windows lets a second copy share a taken port, silently
    try:
        srv = Local(("127.0.0.1", PORT), Handler)
    except OSError as e:
        logger.error(f"port {PORT} on 127.0.0.1 is taken: {e}")
        if _START.get("share"):
            print(f"\nAudio8 could NOT start: port {PORT} on this computer is taken, most likely by\n"
                  "Sapphire's own Audio8 here. In Sapphire: Settings > Audio8, tick Client only\n"
                  "(or turn Audio8 off there), then run this again.\n", flush=True)
        raise SystemExit(1)
    threading.Thread(target=_load_model, daemon=True).start()
    logger.info(f"Audio8 server on 127.0.0.1:{PORT}, voices in {VOICES_DIR}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
