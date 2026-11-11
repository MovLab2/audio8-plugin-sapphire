"""Audio8 TTS provider — thin HTTP client for the plugin's model server,
this computer's or (Client only) another computer's (link.py).

Loaded via exec() by the plugin loader (like piper), so imports are absolute
and everything heavy lives in the service process (server.py, own conda env).
This side only needs requests + a directory scan.
"""
import logging
import re
import time
from pathlib import Path
from typing import Iterator, List, Optional

import requests

from plugins.audio8 import link
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

    @property
    def remote(self) -> bool:
        """Made on another computer (Client only): never on the GPU a face renders on."""
        return bool(link.worker())

    def __init__(self):
        self._health_why = None     # why is_available() said no, in words (the provider switch shows it)
        self._ok_at = 0.0           # when it last answered: one check per few seconds, not per speak
        self._for = ""              # the pairing string _ok_at was for ("" = this computer)
        logger.info("Audio8 TTS provider initialized")

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
            r = link.tts("/tts", self._payload(text, name, speed, kwargs), timeout=(5, 300))
            if r.status_code != 200:
                detail = (r.text or "")[:200]
                logger.error(f"[audio8] server {r.status_code} (voice={name}): {detail}")
                return None
            return r.content or None
        except (requests.RequestException, ValueError) as e:
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
            r = link.tts("/tts/stream", self._payload(text, name, speed, kwargs),
                         stream=True, timeout=(5, 300))
            if r.status_code != 200:
                logger.error(f"[audio8] stream {r.status_code} (voice={name}): {(r.text or '')[:200]}")
                if r.status_code >= 500 and r.status_code != 503:     # 503 = still starting: asking /tts waits it out again
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
        except (requests.RequestException, ValueError) as e:
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
            link.watch()                # the GPU computer's start, as a notification row
        except Exception as e:          # a core before 2026-11-11 (no Watch fetch=): no row, nothing else lost
            logger.debug(f"[audio8] no start row for the GPU computer: {e!r}")
        if link.worker() != self._for:  # Client only flipped, or a new pairing: the last answer was someone else's
            self._for, self._ok_at = link.worker(), 0.0
        if time.monotonic() - self._ok_at < 10:
            return True
        try:
            r = link.call("get", "/health", timeout=(5, 5))
        except Exception as e:
            self._health_why = link.words(e)
            return False
        if r.status_code == 401:
            self._health_why = link.REFUSED
            return False
        if r.status_code != 200:
            self._health_why = f"HTTP {r.status_code}"
            return False
        try:
            h = r.json()
        except ValueError:
            h = {}
        if h.get("status") == "error":
            self._health_why = f"Audio8 failed to start: {h.get('error') or 'see its window or log'}"
            return False
        if self.remote and h.get("status") != "ready":     # its first run downloads and compiles for minutes
            self._health_why = f"Audio8 on the GPU computer is still starting: {h.get('phase') or h.get('status') or 'starting'}"
            return False
        self._health_why = None
        self._ok_at = time.monotonic()
        return True

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
