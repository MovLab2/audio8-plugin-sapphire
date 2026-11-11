"""Audio8 voice library: list, play, upload, transcribe, edit, delete, try.

A voice is <name>.wav + <name>.txt (the exact words). Built-ins ship in the
plugin's voices/ folder and are read-only; the user's live in
user/audio8-voices/. Uploads are converted to 44.1 kHz mono 16-bit WAV here,
so the model server only ever reads one format.

Transcripts come from Sapphire's own STT (the live whisper_client), called
directly: the core /api/transcribe route would fire post_stt and stt_audio
hooks as if the user had spoken. A cloud STT gets a clip only on an explicit ask.
"""
import asyncio
import logging
import os
import re
import tempfile
from math import gcd
from pathlib import Path

logger = logging.getLogger("audio8.voices")

import core.plugin_loader as _pl

BUILTIN_DIR = Path(__file__).absolute().parent.parent / "voices"
# Root from a core module, not this file: a symlinked plugin lives elsewhere.
USER_DIR = Path(_pl.__file__).absolute().parents[1] / "user" / "audio8-voices"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
RATE = 44100
MIN_SECONDS, MAX_SECONDS = 3.0, 30.0
MAX_BYTES = 25 * 1024 * 1024
MAX_SAMPLE_CHARS = 2000
TRY_TEXT = "Hi there. This is how I sound with this voice. Does it feel right to you?"


def _find(name):
    """(wav path, builtin) for a voice name; the user's folder wins."""
    if not name or not NAME_RE.match(name):
        return None, False
    for folder, builtin in ((USER_DIR, False), (BUILTIN_DIR, True)):
        wav = folder / f"{name}.wav"
        if wav.is_file():
            return wav, builtin
    return None, False


def _entry(wav, builtin):
    import soundfile as sf
    txt = wav.with_suffix(".txt")
    try:
        seconds = round(sf.info(str(wav)).duration, 1)
    except Exception:
        seconds = None
    transcript = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.is_file() else ""
    return {"name": wav.stem, "builtin": builtin, "seconds": seconds, "transcript": transcript}


def _stt():
    """(ok, reason, local, provider key) for the live STT."""
    from core.api_fastapi import get_system
    from core.stt.providers import stt_registry
    from core.stt.utils import can_transcribe
    client = getattr(get_system(), "whisper_client", None)
    ok, reason = can_transcribe(client)
    key = stt_registry.get_active_key() or "none"
    try:
        stt_registry.get_all()
        local = bool((stt_registry.get_entry(key) or {}).get("is_local", False))
    except Exception:
        local = False
    return ok, reason, local, key, client


def _transcribe(wav):
    ok, reason, _local, _key, client = _stt()
    if not ok:
        raise RuntimeError(reason)
    # Whisper hears 16 kHz; give it a 16 kHz copy rather than trust each
    # provider to resample a 44.1 kHz file the same way.
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(str(wav), dtype="float32")
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, resample_poly(audio, 16000 // gcd(16000, sr), sr // gcd(16000, sr)), 16000)
        text = client.transcribe_file(tmp)
    finally:
        os.unlink(tmp)
    return " ".join((text or "").split())


def _convert(data, dest):
    """Any libsndfile format (WAV, FLAC, OGG, MP3) -> 44.1k mono PCM16 at dest."""
    import io
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        raise ValueError("Couldn't read that file. Use WAV, MP3, FLAC or OGG.")
    audio = audio.mean(axis=1)
    seconds = len(audio) / sr
    if seconds < MIN_SECONDS:
        raise ValueError(f"That clip is {seconds:.1f}s. Use at least {MIN_SECONDS:.0f}s of speech (5-15s is best).")
    if seconds > MAX_SECONDS:
        raise ValueError(f"That clip is {seconds:.0f}s. Trim it to {MAX_SECONDS:.0f}s or less (5-15s is best).")
    if sr != RATE:
        g = gcd(RATE, sr)
        audio = resample_poly(audio, RATE // g, sr // g)
    peak = float(np.abs(audio).max()) or 1.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(str(dest), audio.astype(np.float32), RATE, subtype="PCM_16")


def list_voices(settings=None, **_):
    voices, seen = [], set()
    for folder, builtin in ((USER_DIR, False), (BUILTIN_DIR, True)):
        if not folder.is_dir():
            continue
        for wav in sorted(folder.glob("*.wav")):
            if NAME_RE.match(wav.stem) and wav.stem not in seen:
                seen.add(wav.stem)
                voices.append(_entry(wav, builtin))
    ok, reason, local, key, _c = _stt()
    return {"voices": voices, "stt": {"ok": ok, "reason": reason, "local": local, "provider": key}}


def voice_audio(name, **_):
    from fastapi.responses import FileResponse
    wav, _b = _find(name)
    if not wav:
        return {"error": "No such voice."}, 404
    return FileResponse(str(wav), media_type="audio/wav")


async def upload_voice(request=None, **_):
    try:
        form = await request.form()
    except Exception:
        return {"error": "Couldn't read the upload."}, 400
    name = str(form.get("name") or "").strip()
    up = form.get("file")
    if not NAME_RE.match(name):
        return {"error": "Name: letters, numbers, - and _ only (up to 40)."}, 400
    if up is None or not hasattr(up, "read"):
        return {"error": "Pick an audio file."}, 400
    wav, builtin = _find(name)
    if wav:
        return {"error": f"'{name}' is taken{' by a built-in voice' if builtin else ''}. Pick another name."}, 409
    data = await up.read()
    if len(data) > MAX_BYTES:
        return {"error": "That file is over 25 MB."}, 413
    USER_DIR.mkdir(parents=True, exist_ok=True)
    dest = USER_DIR / f"{name}.wav"
    try:
        await asyncio.to_thread(_convert, data, dest)
    except ValueError as e:
        return {"error": str(e)}, 400
    note = ""
    ok, reason, local, key, _c = _stt()
    if ok and local:
        try:
            text = await asyncio.to_thread(_transcribe, dest)
            if text:
                dest.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
            else:
                note = "Speech-to-text heard no words. Type what the clip says."
        except Exception as e:
            logger.warning(f"[audio8] transcribe after upload failed: {e!r}")
            note = "Speech-to-text failed. Type what the clip says, or press Transcribe."
    elif ok:
        note = f"Your speech-to-text ({key}) is not local. Press Transcribe to send the clip there, or type the words."
    else:
        note = f"Speech-to-text is unavailable ({reason}). Type what the clip says."
    logger.info(f"[audio8] voice '{name}' added")
    return {"voice": _entry(dest, False), "note": note}


def transcribe_voice(name, body=None, **_):
    wav, _b = _find(name)
    if not wav:
        return {"error": "No such voice."}, 404
    ok, reason, local, key, _c = _stt()
    if not ok:
        return {"error": f"Speech-to-text is unavailable: {reason}"}, 503
    if not local and not (body or {}).get("cloud_ok"):
        return {"error": f"Speech-to-text ({key}) is not local.", "cloud": key}, 409
    try:
        text = _transcribe(wav)
    except Exception as e:
        logger.warning(f"[audio8] transcribe '{name}' failed: {e!r}")
        return {"error": "Speech-to-text failed on that clip."}, 500
    if not text:
        return {"error": "Speech-to-text heard no words in that clip."}, 422
    return {"transcript": text}


def save_transcript(name, body=None, **_):
    wav, builtin = _find(name)
    if not wav:
        return {"error": "No such voice."}, 404
    if builtin:
        return {"error": "Built-in voices are read-only."}, 403
    text = " ".join(str((body or {}).get("transcript") or "").split())
    if not text:
        return {"error": "The transcript can't be empty: the model needs the exact words."}, 400
    wav.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    return {"voice": _entry(wav, False)}


def delete_voice(name, **_):
    wav, builtin = _find(name)
    if not wav:
        return {"error": "No such voice."}, 404
    if builtin:
        return {"error": "Built-in voices can't be deleted."}, 403
    for ext in (".wav", ".txt"):
        p = wav.with_suffix(ext)
        if p.is_file():
            p.unlink()
    logger.info(f"[audio8] voice '{name}' deleted")
    return {"ok": True}


def _mp3(wav_bytes):
    """Lossless server WAV -> MP3 (libsndfile's LAME), for a file to share."""
    import io
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="MP3", subtype="MPEG_LAYER_III")
    return buf.getvalue()


def _num(value, lo, hi):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(lo, min(hi, v)) if v == v else 1.0


def server_status(settings=None, **_):
    """The model server's /health for the page: status, phase, progress, error.
    status "down" = no answer (plugin off, the process starting, or with
    Client only on, the other computer not answering)."""
    from plugins.audio8 import link
    s = settings or {}
    client = link.client_on(s)
    down = {"status": "down", "progress": None, "error": None, "client": client}
    if not client:
        blocked = ((_pl.plugin_loader.get_plugin_info("audio8") or {}).get("services_blocked"))
        if blocked and blocked != "client":
            why = {"missing": "Audio8's environment isn't built: Settings > Plugins > Audio8, Build. "
                              "Or tick Client only to use Audio8 on another computer."}.get(
                blocked, f"Audio8's server can't start here (environment {blocked}): see Settings > Plugins > Audio8")
            return {**down, "phase": why}
    try:
        r = link.call("get", "/health", s, timeout=(5, 5) if client else 2)
    except Exception as e:
        return {**down, "phase": link.words(e) if client else "Starting the server"}
    if r.status_code == 401:
        return {**down, "phase": link.REFUSED}
    try:
        h = r.json() if r.status_code == 200 else {}
    except ValueError:
        h = {}
    if not h:
        return {**down, "phase": f"The Audio8 server answered HTTP {r.status_code}"}
    return {**{k: h.get(k) for k in ("status", "phase", "progress", "error")}, "client": client}


def try_voice(name, body=None, settings=None, **_):
    """Speak text with this voice. Body {text?, speed?, pitch?, format?}:
    no text = a stock test line; format "mp3" = a file to download and share,
    else OGG for a quick listen."""
    from fastapi.responses import Response
    from plugins.audio8 import link
    body = body or {}
    wav, _b = _find(name)
    if not wav:
        return {"error": "No such voice."}, 404
    try:
        temperature = float((settings or {}).get("temperature") or 0.8)
    except (TypeError, ValueError):
        temperature = 0.8
    text = " ".join(str(body.get("text") or "").split())
    if len(text) > MAX_SAMPLE_CHARS:
        return {"error": f"Keep it under {MAX_SAMPLE_CHARS} characters."}, 400
    mp3 = body.get("format") == "mp3"
    try:
        r = link.tts("/tts", {"text": text or TRY_TEXT, "voice": name, "temperature": temperature,
                              "speed": _num(body.get("speed", 1.0), 0.75, 1.6),
                              "pitch": _num(body.get("pitch", 1.0), 0.5, 2.0),
                              "format": "wav" if mp3 else "ogg"}, settings or None, timeout=(5, 300))
    except Exception as e:
        if link.client_on(settings or None):
            return {"error": f"Audio8 on the other computer: {link.words(e)}"}, 503
        return {"error": "The Audio8 server isn't answering. Is the plugin on and loaded?"}, 503
    if r.status_code != 200:
        if r.status_code == 401:
            words = link.REFUSED
        else:
            try:
                words = r.json().get("error")
            except Exception:
                words = r.text
        # The worker's own codes stay there: a 401 here would read as Sapphire's login expiring.
        code = 502 if link.client_on(settings or None) and r.status_code not in (400, 503) else r.status_code
        return {"error": str(words or "The Audio8 server refused.")[:300]}, code
    if mp3:
        return Response(_mp3(r.content), media_type="audio/mpeg")
    return Response(r.content, media_type="audio/ogg")
