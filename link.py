"""How Sapphire reaches the Audio8 server: this computer's (127.0.0.1:port),
or with `client` on, the server on another computer named by the pairing
string in `worker` (core/worker_door.py: pinned TLS, a key per computer).

A worker has none of this computer's voices, so a voice goes by the hash of
its clip and transcript (server.voice_hash); the first time a worker answers
409 the clip is sent once (POST /voice) and the request is asked again.
Shared by provider.py and routes/voices.py.
"""
import base64
import hashlib
import logging
import re
import socket
import ssl
import threading
import time
from pathlib import Path

import requests

from core import net, worker_door
import core.plugin_loader as _pl

logger = logging.getLogger("audio8.link")

ROOT = Path(_pl.__file__).absolute().parents[1]          # never resolve(): a symlinked plugin
VOICES_DIR = ROOT / "user" / "audio8-voices"
BUILTIN_DIR = Path(__file__).absolute().parent / "voices"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

DOWN_S = 5.0            # a worker that just failed is not asked again (a ~5 s cert fetch) for this long

_lock = threading.Lock()
_doors = {}              # pairing string -> (session, base)
_down = {}               # pairing string -> (until, words)
_hashes = {}             # wav path -> ((wav mtime, txt mtime), hash name)


def settings() -> dict:
    try:
        return _pl.plugin_loader.get_plugin_settings("audio8") or {}
    except Exception:
        return {}


class Unreachable(requests.ConnectionError):
    """The other computer didn't answer; str() is the why, in words."""


REFUSED = ("the GPU computer refused this computer's key (it was removed there, "
           "or this pairing string is old): copy its pairing string again")


def client_on(s=None) -> bool:
    s = settings() if s is None else s
    return s.get("client") is True


def worker(s=None) -> str:
    """The pairing string when Client only is on, else ""."""
    s = settings() if s is None else s
    return str(s.get("worker") or "").strip() if client_on(s) else ""


def words(e) -> str:
    """Why a call to the other computer failed, for the page and the log."""
    t = str(e)
    if isinstance(e, (ValueError, Unreachable)) or "certificate changed" in t:
        return t
    low = t.lower()
    if isinstance(e, (requests.Timeout, socket.timeout, TimeoutError)) or "timed out" in low:
        return "no answer in time: is Audio8 running there, and its port allowed in that computer's firewall?"
    if isinstance(e, ConnectionRefusedError) or "refused" in low:
        return ("nothing answers at that address: Audio8 isn't running there, or that computer's "
                "address changed (copy its pairing string again)")
    if isinstance(e, socket.gaierror) or "name or service" in low or "getaddrinfo" in low:
        return "that computer's address can't be found"
    if isinstance(e, ssl.SSLError):
        return f"the secure connection failed ({type(e).__name__})"
    return f"no answer from the other computer ({type(e).__name__})"


def call(method, path, s=None, **kw):
    """One request to the server: here through the net facade, on a worker
    through its pinned door. Client only on with no usable pairing string is
    a ValueError, an unreachable worker is Unreachable: both with words."""
    s = settings() if s is None else s
    if not client_on(s):
        try:
            port = int(s.get("port") or 5014)
        except (TypeError, ValueError):
            port = 5014
        return getattr(net, method)(f"http://127.0.0.1:{port}{path}", **kw)
    pairing = worker(s)
    if not pairing:
        raise ValueError("Client only is on, but no pairing string is pasted (Settings > Audio8)")
    if not worker_door.is_pairing(pairing):
        raise ValueError("Client only is on, but that isn't a pairing string (https://host:port/#key=...&pin=...)")
    with _lock:
        door, down = _doors.get(pairing), _down.get(pairing)
    if down and down[0] > time.monotonic() and method.lower() == "get":
        raise Unreachable(down[1])                      # a look (health, status); her words always try the door: a Wi-Fi blip dropped paragraphs
    try:
        if not door:
            door = worker_door.session(pairing, timeout=5)
            with _lock:
                _doors[pairing] = door
        sess, base = door
        r = sess.request(method.upper(), f"{base}{path}", **kw)
    except (requests.exceptions.SSLError, ssl.SSLError) as e:
        with _lock:
            _doors.pop(pairing, None)                   # a new cert there: checked afresh, and refused
        worker_door.forget(*worker_door.parse(pairing)[::2])
        _mark_down(pairing, e)
    except (requests.RequestException, OSError) as e:
        _mark_down(pairing, e)
    with _lock:
        _down.pop(pairing, None)
    return r


def _mark_down(pairing, e):
    why = words(e)
    with _lock:
        _down[pairing] = (time.monotonic() + DOWN_S, why)
    raise Unreachable(why) from e


_watch = {"pairing": "", "w": None}


def watch():
    """The other computer's start as a notification row, like this computer's
    own server gets (core/service_health): a first run there downloads and
    compiles for minutes, and nothing here said so (scouts 2026-11-10). One
    watch per pairing string; none while Client only is off."""
    pairing = worker()
    with _lock:
        if _watch["pairing"] == pairing:
            return
        old = _watch["w"]
        _watch.update(pairing=pairing, w=None)
    if old:
        old.stop()
    if pairing:
        from core import service_health
        w = service_health.Watch("audio8", "worker", "Audio8 (GPU computer)", None, lambda: True, fetch=_health)
        with _lock:
            _watch["w"] = w


def _health():
    try:
        return call("get", "/health", timeout=(5, 5)).json()
    except (requests.RequestException, ValueError) as e:
        return {"status": "starting", "phase": words(e) if isinstance(e, requests.RequestException) else str(e)}


def _clip(name):
    """(wav, txt) paths of a voice here, user's first; None if it has no clip."""
    for d in (VOICES_DIR, BUILTIN_DIR):
        wav = d / f"{name}.wav"
        if wav.is_file():
            return wav, wav.with_suffix(".txt")
    return None


def _hash_name(wav, txt):
    try:
        stamp = (wav.stat().st_mtime, txt.stat().st_mtime)
    except OSError:
        return None                                     # no transcript: the worker says what it says for any
    with _lock:
        hit = _hashes.get(wav)
    if hit and hit[0] == stamp:
        return hit[1]
    text = txt.read_text(encoding="utf-8", errors="replace").strip()
    name = "h-" + hashlib.sha256(wav.read_bytes() + b"\0" + text.encode("utf-8")).hexdigest()
    with _lock:
        _hashes[wav] = (stamp, name)
    return name


def _send(wav, txt, s):
    body = {"wav": base64.b64encode(wav.read_bytes()).decode("ascii"),
            "text": txt.read_text(encoding="utf-8", errors="replace").strip()}
    r = call("post", "/voice", s, json=body, timeout=60)
    if r.status_code != 200:
        raise requests.RequestException(f"the worker refused the voice clip ({r.status_code}): {r.text[:200]}")
    logger.info(f"[audio8] sent voice {wav.stem} to the worker")


def tts(path, payload, s=None, **kw):
    """POST /tts or /tts/stream. On a worker the voice goes by its hash, and
    a 409 (not there yet) sends the clip once and asks again."""
    s = settings() if s is None else s
    if not worker(s):
        return call("post", path, s, json=payload, **kw)
    name = payload.get("voice") or ""
    clip = _clip(name) if NAME_RE.match(name) else None
    hashed = _hash_name(*clip) if clip else None
    if clip and not hashed:                             # as this computer's server says it
        raise ValueError(f"voice '{name}' has no transcript: add {name}.txt with the exact words spoken in the wav")
    body = dict(payload, voice=hashed) if hashed else payload
    r = call("post", path, s, json=body, **kw)
    if r.status_code == 409 and hashed:
        r.close()
        _send(*clip, s)
        r = call("post", path, s, json=body, **kw)
    return r
