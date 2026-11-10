"""Audio8 TTS provider — thin HTTP client for the plugin's model server.

Loaded via exec() by the plugin loader (like piper), so imports are absolute
and everything heavy lives in the service process (server.py, own conda env).
This side only needs requests + a directory scan.
"""
import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional

import requests

from core import net

from core.tts.providers.base import BaseTTSProvider
import core.tts.providers.base as _base

logger = logging.getLogger(__name__)

# Anchor to the CORE package so symlinked plugins still resolve the real root
# (.absolute(), never .resolve(): resolve() follows a symlink out of the Sapphire tree).
SAPPHIRE_ROOT = Path(_base.__file__).absolute().parents[3]
VOICES_DIR = SAPPHIRE_ROOT / "user" / "audio8-voices"
BUILTIN_DIR = Path(__file__).absolute().parent / "voices"   # read-only, ships with the plugin
VOICE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _settings() -> dict:
    try:
        from core.plugin_loader import plugin_loader
        return plugin_loader.get_plugin_settings("audio8") or {}
    except Exception:
        return {}


class Audio8TTSProvider(BaseTTSProvider):
    """Posts text to the local Audio8 server, returns one OGG/Opus blob."""

    audio_content_type = "audio/ogg"
    # Speed = a time-stretched reference the model copies; it follows well
    # in this band (server clamps the same). Pitch = varispeed, server-side.
    SPEED_MIN = 0.75
    SPEED_MAX = 1.6
    supports_pitch = True
    # The core pump chunks the reply and calls generate() per chunk via the
    # base generate_stream(); the server's GPU lock serializes its workers.
    supports_streaming = True
    # One GPU lock on the server: a second pump worker only races chunk N for
    # it (an unfair lock once let N+1 go first and N timed out, 2026-11-08).
    synth_workers = 1

    def __init__(self):
        logger.info("Audio8 TTS provider initialized")

    @property
    def _base_url(self) -> str:
        try:
            port = int(_settings().get("port") or 5014)
        except (TypeError, ValueError):
            port = 5014
        return f"http://127.0.0.1:{port}"

    def _resolve_voice(self, voice: str) -> str:
        """Use the chat-passed voice only if it's one of our wavs (e.g. an
        af_heart Kokoro leftover isn't); else the plugin's default setting."""
        if voice and VOICE_RE.match(voice) and any(
                (d / f"{voice}.wav").is_file() for d in (VOICES_DIR, BUILTIN_DIR)):
            return voice
        return (_settings().get("voice") or "").strip() or "sapphire"

    def generate(self, text: str, voice: str, speed: float, **kwargs) -> Optional[bytes]:
        if not text or not text.strip():
            return None
        name = self._resolve_voice(voice)
        try:
            r = net.post(f"{self._base_url}/tts", json=self._payload(text, name, speed, kwargs),
                         timeout=300)
            if r.status_code != 200:
                detail = (r.text or "")[:200]
                logger.error(f"[audio8] server {r.status_code} (voice={name}): {detail}")
                return None
            return r.content or None
        except requests.RequestException as e:
            logger.error(f"[audio8] request failed (voice={name}): {e!r}")
            return None

    def _payload(self, text, name, speed, kwargs) -> dict:
        try:
            temperature = float(_settings().get("temperature") or 0.8)
        except (TypeError, ValueError):
            temperature = 0.8
        return {"text": text, "voice": name, "temperature": temperature,
                "pieces": _settings().get("pieces") is True,
                "speed": max(self.SPEED_MIN, min(self.SPEED_MAX, float(speed or 1.0))),
                "pitch": float(kwargs.get("pitch") or 1.0)}

    def generate_stream(self, text: str, voice: str, speed: float, **kwargs) -> Iterator[bytes]:
        """Yield each OGG segment as the server makes it (cut at pauses, about
        every 1.5-6 s of speech), so the first audio comes within a second
        instead of after the whole chunk and the pump sees steady progress. Falls back to one blob on a server error
        before any audio; a 4xx (bad voice) would fail the same way, so not."""
        if not text or not text.strip():
            return
        from core.tts.providers.kokoro import _split_complete_oggs
        name = self._resolve_voice(voice)
        sent = 0
        r = None
        try:
            r = net.post(f"{self._base_url}/tts/stream", json=self._payload(text, name, speed, kwargs),
                         stream=True, timeout=(5, 300))
            if r.status_code != 200:
                logger.error(f"[audio8] stream {r.status_code} (voice={name}): {(r.text or '')[:200]}")
                if r.status_code >= 500:
                    audio = self.generate(text, voice, speed, **kwargs)
                    if audio:
                        yield audio
                return
            buf = b""
            for part in r.iter_content(chunk_size=None):
                if not part:
                    continue
                files, buf = _split_complete_oggs(buf + part)
                for ogg in files:
                    sent += 1
                    yield ogg
            if buf:
                sent += 1
                yield buf
        except requests.RequestException as e:
            logger.error(f"[audio8] stream failed after {sent} segment(s) (voice={name}): {e!r}")
            if sent == 0:
                audio = self.generate(text, voice, speed, **kwargs)
                if audio:
                    yield audio
        finally:
            # Closing mid-body (a Stop, an abandoned stream) drops the socket;
            # the server sees the broken pipe and frees its GPU lock.
            if r is not None:
                r.close()

    def is_available(self) -> bool:
        try:
            r = net.get(f"{self._base_url}/health", timeout=2)
            return r.status_code == 200 and r.json().get("status") != "error"
        except Exception:
            return False

    def list_voices(self) -> List[dict]:
        """One entry per wav in user/audio8-voices/ then the built-ins (a
        user voice shadows a built-in of the same name); transcript-less wavs
        are flagged in the label (generation refuses them with a clear error)."""
        voices, seen = [], set()
        try:
            for wav in [*sorted(VOICES_DIR.glob("*.wav")), *sorted(BUILTIN_DIR.glob("*.wav"))]:
                stem = wav.stem
                if not VOICE_RE.match(stem) or stem in seen:
                    continue
                seen.add(stem)
                has_txt = wav.with_suffix(".txt").is_file()
                voices.append({
                    "voice_id": stem,
                    "name": stem if has_txt else f"{stem} (needs .txt transcript)",
                    "category": "Audio8",
                    "downloaded": True,
                })
        except OSError as e:
            logger.error(f"[audio8] voice scan failed: {e!r}")
        return voices
