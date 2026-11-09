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
compiled -> graph -> original model.generate; a rung that fails to warm up
drops to the next, and a runtime failure demotes to original for good.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import os
import re
import sys
import threading

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
# torch.compile's kernel caches: rebuildable, so under user/models/ (never
# backed up). Cold compile ~40s, warm ~15s; without a cache, every start pays 40.
_CACHE = os.path.join(ROOT, "user", "models", "audio8")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE, "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE, "triton"))
VOICE_RE = re.compile(r"^[A-Za-z0-9_-]+$")  # blocks path traversal via voice name
MAX_GROUP_CHARS = 300
SPEED_MIN, SPEED_MAX = 0.75, 1.6   # the model follows a stretched reference well in this band
PITCH_MIN, PITCH_MAX = 0.5, 2.0
READY_TIMEOUT = 600  # first-ever load may download the model

_state = {"status": "loading", "error": None, "engine": None}
_ready = threading.Event()
_gpu_lock = threading.Lock()
_model = None
_processor = None
_device = "cpu"
_fast = None               # FastGen, or None on the original engine
_ref_cache = {}            # (wav, mtime, speed) -> {"array", "sampling_rate"}
_ref_lock = threading.Lock()
_REF_CACHE_MAX = 32


def _load_model():
    global _model, _processor, _device
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if _device == "cuda" else torch.float32
        logger.info(f"Loading {MODEL_ID}@{MODEL_REV[:8]} on {_device}")
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REV, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REV, trust_remote_code=True, dtype=dtype
        ).eval().to(_device)
        _warm_engine()
        _state["status"] = "ready"
        logger.info(f"Model ready (engine: {_state['engine']})")
    except Exception as e:
        _state.update(status="error", error=str(e))
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
    for name in ("compiled", "graph"):
        t = time.time()
        try:
            fg = FastGen(_model)
            if name == "compiled":
                fg._frame = torch.compile(fg._frame, dynamic=False)
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


def _resolve_voice(name):
    """Return (wav_path, transcript) or (None, None) for base voice."""
    if not name or not VOICE_RE.match(name):
        return None, None
    folder = next((d for d in (VOICES_DIR, BUILTIN_DIR)
                   if os.path.isfile(os.path.join(d, f"{name}.wav"))), None)
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


def _render(text, voice, temperature, speed=1.0, pitch=1.0):
    """Yield float32 audio per sentence group (~MAX_GROUP_CHARS), at the codec
    rate, each group after the first led by a short breath gap. Voice and
    reference problems raise ValueError BEFORE the first yield, so a caller
    can still answer 400. The GPU lock spans the whole request: one text's
    groups never interleave with another request's."""
    import numpy as np
    import torch

    wav_path, ref_text = _resolve_voice(voice)
    ref_audio = _reference(wav_path, speed) if wav_path else None
    sr = _model.config.codec_sample_rate
    gap = np.zeros(int(0.12 * sr), dtype="float32")

    def groups():
        with _gpu_lock:
            for n, chunk in enumerate(_split_text(text)):
                kwargs = {"text": [chunk], "return_tensors": "pt"}
                if wav_path:
                    kwargs["reference_audio"] = [ref_audio]
                    kwargs["reference_text"] = [ref_text]
                inputs = _processor(**kwargs)
                inputs = {k: v.to(_device) for k, v in inputs.items()}
                with torch.inference_mode():
                    waveforms, lengths = _model.decode_audio(_codes(inputs, temperature))
                audio = waveforms[0, : int(lengths[0])].float().cpu().numpy()
                yield np.concatenate([gap, audio]) if n else audio
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
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, _state)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/tts", "/tts/stream"):
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._json(400, {"error": "bad json"})
            return
        text = (data.get("text") or "").strip()
        if not text:
            self._json(400, {"error": "empty text"})
            return
        if not _ready.wait(timeout=READY_TIMEOUT) or _state["status"] != "ready":
            self._json(503, {"error": f"model not ready: {_state}"})
            return
        temperature = _clamp(data.get("temperature") or 0.8, 0.05, 2.0, 0.8)
        speed = _clamp(data.get("speed", 1.0), SPEED_MIN, SPEED_MAX)
        pitch = _clamp(data.get("pitch", 1.0), PITCH_MIN, PITCH_MAX)
        if self.path == "/tts/stream":
            self._stream(text, data.get("voice"), temperature, speed, pitch)
            return
        try:
            fmt = "wav" if data.get("format") == "wav" else "ogg"
            audio = _synth(text, data.get("voice"), temperature, speed, pitch, fmt)
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

    def _stream(self, text, voice, temperature, speed, pitch):
        """One complete OGG per sentence group, written the moment it's made
        (Kokoro's /tts/stream shape; the provider splits files at EOS pages).
        Chunked transfer, one HTTP chunk per OGG: a close-delimited body made
        requests' iter_content wait for the END before yielding anything.
        A failure after the 200 can only end the body early."""
        try:
            groups = _render(text, voice, temperature, speed, pitch)
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
        except (BrokenPipeError, ConnectionResetError):
            logger.info("stream client went away")       # a Stop: the pump closed it
        except Exception as e:
            logger.error(f"stream synthesis failed: {e!r}")
        finally:
            groups.close()                                 # releases the GPU lock now


def main():
    os.makedirs(VOICES_DIR, exist_ok=True)
    threading.Thread(target=_load_model, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    logger.info(f"Audio8 server on 127.0.0.1:{PORT}, voices in {VOICES_DIR}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
