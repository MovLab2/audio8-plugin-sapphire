# Audio8 TTS

Zero-shot voice-cloning TTS at 44.1kHz — [Audio8-TTS-Realistic-Air](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air)
(601M DualAR, 23 languages, Apache-2.0), pinned to a commit sha. A finetune of
Audio8-TTS-Preview-0.6b with more natural prosody; same architecture and code. ~1.2GB VRAM, ~11×
realtime on an RTX 5090 with the built-in fast engine (~2× on the model's own loop).

## How it runs

- **Own conda env** (`sapphire-plugin-audio8`) — build it from Settings →
  Plugins → Audio8 (one click, ~3.5GB torch download). Keeps the model's
  `transformers<5` pin away from the main install.
- **Model server** (`server.py`) runs as a plugin service on port 5014
  (setting). First start downloads ~2.5GB of weights into the HF cache.
- **Provider** (`provider.py`) is a thin HTTP client registered as TTS
  provider `audio8` — select it in Settings → TTS.

## Speed engine

`fastgen.py` replays each decode frame as one CUDA graph, fused by
`torch.compile` with kernels autotuned for your GPU (~15x realtime on a 5090,
vs ~2x for the model's own loop). The server picks the best engine that warms
up, logged and shown in `/health` as `engine`: `tuned` -> `compiled` (~11x)
-> `graph` (no Triton, e.g. Windows, ~5.7x) -> `original`. Warm-up runs
before the server reports ready: a few minutes the first time (it tunes
kernels on your card), ~20s after (kernel cache in `user/models/audio8/`, never backed up).
After a model revision bump, run `tools/parity.py` with the env's python: it
checks the fast loop still decodes like the model's own (greedy decoding
identical, filter and RAS matching, sampler exact).

## Voices

Settings > Audio8 is the voice library: play each clip, try a test line,
add a clip, fix its transcript, delete your own. A new clip is converted to
44.1kHz mono WAV and transcribed by Sapphire's own speech-to-text (a cloud
STT asks first); fix any wrong words and Save, since the model needs them exact.

The **Sampler** at the top speaks anything you type (up to 2000 characters)
with any voice, speed and pitch, and **Download MP3** saves the take (encoded
from the server's lossless WAV, not from Opus).

A voice is a pair:

- `<name>.wav`: 5-15s of clean, quiet speech, one speaker (3-30s accepted)
- `<name>.txt`: the **exact** words spoken in the wav

Built-in voices ship read-only in the plugin's `voices/` folder: `sapphire`
is Kokoro `af_heart` at speed 1.2, pitch 0.96 (Kokoro and its voices are
Apache-2.0). Your own live in `user/audio8-voices/`; a voice of yours with a
built-in's name wins. All of them appear in the chat voice dropdown; a wav
without its `.txt` shows "(needs .txt transcript)" and refuses to generate.

The default voice setting is used when a chat's voice isn't an Audio8 voice
(e.g. a leftover Kokoro name).

## Audio8 on another computer

Run the voice on a second computer's graphics card, so Sapphire's computer
keeps its own card free (for her face, for example). The second computer needs
an NVIDIA graphics card: on a processor alone Audio8 is far too slow to talk.

**On the second computer (once):**

1. Install Miniconda: https://www.anaconda.com/download/success (the
   Miniconda installer, default options).
2. Download this plugin (on its GitHub page: **Code > Download ZIP**) and
   unzip it. Inside is one folder (named like `audio8-main`): move or rename
   it so the files sit right in a short folder, like `C:\audio8`. Check: you
   should see `audio8-server.bat` when you open `C:\audio8`.

**On the second computer (every time):**

3. Open **Anaconda Prompt** from the Start menu.
4. Type these two lines, pressing Enter after each:

   ```
   cd C:\audio8
   audio8-server.bat
   ```

   The first time takes a while: it downloads about 7 GB and sets itself up.
   If Windows asks whether Python may use the network, tick **Private
   networks** and click **Allow** (it asks again after the first run builds
   its own Python, even if FlashHead was allowed before).
5. A box appears with a long line starting `https://`. That's the pairing
   string: select it and copy it (it's the same every time, so you only paste
   it once). When the window says **Model ready**, Audio8 is ready.
6. Leave that window open. Closing it turns Audio8 off.

**In Sapphire (once):**

7. Settings > Audio8: tick **Client only**, paste the pairing string into
   the box under it, and click **Save**.

That's it. Sapphire now speaks with the other computer's graphics card. Your
voices stay on Sapphire's computer and are sent over by themselves the first
time each one is used.

**If it stops working:** Settings > Audio8 says what's wrong at the top.

- Check that the window from step 6 is still open and says Model ready.
- If the second computer got a new network address (after a router restart,
  say), the pairing string changes: copy the new one from the window and
  paste it in again (step 7).
- "No answer in time" usually means the firewall. On the second computer,
  open **Windows Security > Firewall & network protection > Allow an app
  through firewall**, find **python** and tick **Private**. Also check that
  your home network is set to **Private**, not Public (Settings > Network &
  internet > your connection).
- If the window says it could NOT open the port, Audio8 is already running
  in another window: use that one.

On Linux, step 4 is `./audio8-server.sh` in a terminal. To stop using the
other computer, untick Client only and Save.

*For the technical:* the door is TLS on port 5015 (`set PORT=...` before the
launcher to change it), pinned to that computer's certificate, with a key per
computer. `audio8-server.bat add NAME` makes a pairing string for another
Sapphire, `remove NAME` shuts one out, `list` shows who came in and when.
Voices are kept there by the hash of clip + transcript (newest 64), so an
edited voice is re-sent by itself.

## Knobs & limits

- `temperature` (default 0.8) — every generation is a fresh sampled take.
- **Speed** (0.75–1.6) — the model has no rate knob but copies the
  reference clip's pace, so the server time-stretches the reference
  (rubberband via `pedalboard`, cached per voice and speed) and the model
  speaks at that pace natively. The base voice (no reference) ignores speed.
- **Pitch** — varispeed like Kokoro/Piper (lower = deeper and slightly slower),
  applied server-side in the Opus resample.
- Streaming: core's pump sends the reply in chunks as it is written; each
  chunk is sentence-batched server-side (~300 chars per generate).

## Responsible use

Only clone a voice you have permission to clone, and say the audio is
synthetic where that matters; the model's authors ask the same ("Obtain
permission to clone a voice and disclose synthetic audio where appropriate").
The built-in `sapphire` voice is itself synthetic (Kokoro `af_heart`), not a
recording of any person. Generated audio is not watermarked.

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`). Unofficial: not made or
endorsed by Edge0. The plugin runs inside Sapphire (AGPL-3.0) and imports its
core; this plugin's own code is Apache-2.0.

The plugin bundles no model or weights: the model (Apache-2.0 per Edge0)
downloads from Hugging Face at a pinned revision, and its libraries install
into the plugin's own environment (torch and torchaudio BSD, transformers
and safetensors Apache-2.0, soundfile BSD, pedalboard GPL-3.0; the
standalone launcher adds cryptography, Apache-2.0/BSD).

`fastgen.py` (the fast decode loop) drives the model's own modules at run
time. Its loop is modelled on the model's `generate()` (Apache-2.0, Edge0;
see `NOTICE`) and its sampling is written from the published methods: top-k
and nucleus sampling, Gumbel-max and RAS (VALL-E 2). `tools/parity.py`
checks it decodes like the model. The built-in `sapphire` voice was
generated with Kokoro-82M (Apache-2.0).
