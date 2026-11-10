// Audio8 settings: the three plain settings (global Save) and a voice
// library that saves with its own buttons (upload, transcribe, edit, delete).

import { registerPluginSettings } from '/static/shared/plugin-registry.js';
import pluginsAPI from '/static/shared/plugins-api.js';
import { fetchWithTimeout, errWords } from '/static/shared/fetch.js';
import { escapeHtml, showConfirm } from '/static/shared/modal.js';
import { flash } from '/static/shared/notify.js';
import { saveBlob } from '/static/shared/download.js';

const API = '/api/plugin/audio8/voices';
const vurl = (name, tail = '') => `${API}/${encodeURIComponent(name)}${tail}`;

function injectStyles() {
    if (document.getElementById('a8-styles')) return;
    const s = document.createElement('style');
    s.id = 'a8-styles';
    s.textContent = `
        .a8 { display: flex; flex-direction: column; gap: var(--space-lg); }
        .a8 h4 { margin: 0 0 var(--space-sm); font-size: var(--font-md); }
        .a8-row { display: flex; gap: var(--space-sm); align-items: center; flex-wrap: wrap; }
        .a8-field { display: flex; flex-direction: column; gap: 4px; margin-bottom: var(--space-md); }
        .a8-field label { font-weight: 600; font-size: var(--font-base); }
        .a8-hint { color: var(--text-muted); font-size: var(--font-sm); line-height: 1.5; }
        .a8 input[type=text], .a8 input[type=number], .a8 select, .a8 textarea {
            background: var(--bg-tertiary); color: var(--text); border: 1px solid var(--border);
            border-radius: var(--radius-sm); padding: 6px 8px; font: inherit; }
        .a8 textarea { width: 100%; box-sizing: border-box; resize: vertical; min-height: 54px; }
        .a8 textarea[readonly] { opacity: .75; }
        .a8-card { background: var(--bg-secondary); border: 1px solid var(--border);
            border-radius: var(--radius-md); padding: var(--space-md); display: flex;
            flex-direction: column; gap: var(--space-sm); }
        .a8-name { font-weight: 600; font-size: var(--font-md); }
        .a8-badge { font-size: var(--font-sm); color: var(--text-muted); border: 1px solid var(--border-light);
            border-radius: 10px; padding: 1px 8px; }
        .a8-badge.warn { color: var(--warning); border-color: var(--warning); }
        .a8 button { background: var(--bg-tertiary); color: var(--text); border: 1px solid var(--border-light);
            border-radius: var(--radius-sm); padding: 5px 10px; cursor: pointer; font: inherit; }
        .a8 button:hover:not(:disabled) { background: var(--bg-hover); }
        .a8 button:disabled { opacity: .5; cursor: default; }
        .a8 button.danger { color: var(--error); border-color: var(--error); }
        .a8 audio { height: 32px; max-width: 100%; }
        .a8-spacer { flex: 1; }
        .a8-hint.err { color: var(--error); }
        .a8 [hidden] { display: none !important; }
        .a8-status { display: flex; flex-direction: column; gap: 6px; border-color: var(--warning); }
        .a8-status.err { border-color: var(--error); }
        .a8-status progress { width: 100%; height: 8px; }
    `;
    document.head.appendChild(s);
}

async function api(url, opts = {}, timeout = 120000) {
    return fetchWithTimeout(url, opts, timeout);
}

function json(method, body) {
    return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) };
}

let lib = { voices: [], stt: {} };
let ready = false;              // the model server answers; Speak and Try wait for it

// Speak and Try stay off until the server says ready (it can take minutes on
// a first start: the model downloads and compiles).
function applyReady(box) {
    box.querySelectorAll('#a8-s-go, [data-act="try"]').forEach(b => {
        if (!b.dataset.busy) b.disabled = !ready;
    });
}

async function watchStatus(box) {
    const el = box.querySelector('#a8-status');
    const me = box._a8watch = {};             // a re-render starts a new watch; this one bows out
    while (box.isConnected && box._a8watch === me) {
        let st;
        try { st = await api('/api/plugin/audio8/status', {}, 5000); }
        catch { st = { status: 'down', phase: 'No answer from Sapphire' }; }
        ready = st.status === 'ready';
        applyReady(box);
        el.hidden = ready;
        if (ready) {                          // keep an eye out: a crash restarts it
            await new Promise(r => setTimeout(r, 5000));
            continue;
        }
        const p = st.progress, bar = el.querySelector('progress');
        el.classList.toggle('err', st.status === 'error');
        el.querySelector('.a8-name').textContent = st.status === 'error'
            ? 'Audio8 failed to start' : 'Audio8 is starting';
        el.querySelector('.a8-hint').textContent = st.status === 'error'
            ? `${st.error || 'see user/logs/audio8-audio8-server.log'}`
            : `${st.phase || 'Starting'}${p?.total ? ` - ${(p.done / 1e9).toFixed(2)} of ${(p.total / 1e9).toFixed(2)} GB` : ''}. Speak and Try turn on when it is ready.`;
        bar.hidden = !p?.total;
        if (p?.total) { bar.max = p.total; bar.value = p.done; }
        await new Promise(r => setTimeout(r, st.status === 'error' ? 5000 : 1500));
    }
}

function renderGeneral(box, settings) {
    const g = box.querySelector('#a8-general');
    g.innerHTML = `
        <div class="a8-field">
            <label for="a8-voice">Default voice</label>
            <select id="a8-voice"></select>
            <div class="a8-hint">Used when the chat's voice isn't an Audio8 voice.</div>
        </div>
        <div class="a8-row">
            <div class="a8-field">
                <label for="a8-temp">Temperature</label>
                <input type="number" id="a8-temp" step="0.05" min="0.1" max="1.5" value="${Number(settings.temperature ?? 0.8)}">
            </div>
            <div class="a8-field">
                <label for="a8-port">Server port</label>
                <input type="number" id="a8-port" min="1024" max="65535" value="${Number(settings.port ?? 5014)}">
            </div>
        </div>
        <div class="a8-hint">Temperature: lower is steadier, higher is livelier (model default 0.8). A port change needs a plugin toggle.</div>
        <div class="a8-field">
            <label><input type="checkbox" id="a8-pieces" ${settings.pieces === true ? 'checked' : ''}> Stream in pieces (experimental)</label>
            <div class="a8-hint">Off is recommended. On sends speech in short pieces as it's made: first words come sooner, but it costs more GPU time and adds small gaps between pieces.</div>
        </div>`;
    g.dataset.voice = settings.voice || 'sapphire';
}

function fillVoiceSelect(box) {
    const sel = box.querySelector('#a8-voice');
    const want = sel.value || box.querySelector('#a8-general').dataset.voice;
    const names = lib.voices.map(v => v.name);
    if (want && !names.includes(want)) names.push(want);
    sel.innerHTML = names.map(n => {
        const v = lib.voices.find(x => x.name === n);
        const tag = !v ? ' (missing)' : (!v.transcript ? ' (needs transcript)' : '');
        return `<option value="${escapeHtml(n)}"${n === want ? ' selected' : ''}>${escapeHtml(n + tag)}</option>`;
    }).join('');
    const pick = box.querySelector('#a8-s-voice');
    const had = pick.value || want;
    const usable = lib.voices.filter(v => v.transcript).map(v => v.name);
    pick.innerHTML = usable.map(n => `<option value="${escapeHtml(n)}"${n === had ? ' selected' : ''}>${escapeHtml(n)}</option>`).join('');
}

function wireSampler(box) {
    const $ = id => box.querySelector(id);
    const go = $('#a8-s-go'), dl = $('#a8-s-dl'), player = $('#a8-s-audio'), text = $('#a8-s-text');
    let last = null;
    const speak = async () => {
        const voice = $('#a8-s-voice').value;
        if (!voice) { flash('Add a voice with a transcript first', 'warning'); return; }
        if (!text.value.trim()) { flash('Type something for the voice to say', 'warning'); text.focus(); return; }
        go.disabled = true; go.dataset.busy = '1'; go.textContent = 'Speaking...';
        try {
            const blob = await api(vurl(voice, '/try'), json('POST', {
                text: text.value, format: 'mp3',
                speed: parseFloat($('#a8-s-speed').value) || 1, pitch: parseFloat($('#a8-s-pitch').value) || 1,
            }), 300000);
            if (player.src) URL.revokeObjectURL(player.src);
            player.src = URL.createObjectURL(blob);
            player.hidden = false;
            last = { blob, voice };
            dl.disabled = false;
            player.play().catch(() => {});
        } catch (e) { flash(errWords(e), 'error'); }
        delete go.dataset.busy; go.disabled = !ready; go.textContent = 'Speak';
    };
    go.addEventListener('click', speak);
    text.addEventListener('keydown', e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); speak(); } });
    text.addEventListener('input', () => { $('#a8-s-count').textContent = `${text.value.length} / 2000`; });
    dl.addEventListener('click', () => {
        if (!last) return;
        const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
        saveBlob(last.blob, `${last.voice}-${stamp}.mp3`);
    });
}

function sttLine() {
    const s = lib.stt || {};
    if (!s.ok) return `Speech-to-text is off (${escapeHtml(s.reason || 'unavailable')}), so type each transcript yourself.`;
    if (!s.local) return `Speech-to-text (${escapeHtml(s.provider)}) is not local: Transcribe asks before sending a clip there.`;
    return `New clips are transcribed by your speech-to-text (${escapeHtml(s.provider)}). Fix any words it got wrong, then Save.`;
}

function card(v, box) {
    const el = document.createElement('div');
    el.className = 'a8-card';
    const badges = [
        v.builtin ? '<span class="a8-badge">built-in</span>' : '',
        v.seconds != null ? `<span class="a8-badge">${v.seconds}s</span>` : '',
        !v.transcript ? '<span class="a8-badge warn">needs transcript</span>' : '',
    ].join('');
    el.innerHTML = `
        <div class="a8-row"><span class="a8-name">${escapeHtml(v.name)}</span>${badges}</div>
        <div class="a8-row">
            <audio controls preload="none" src="${vurl(v.name, '/audio')}"></audio>
            <button type="button" data-act="try" title="Speak a test line with this voice">Try it</button>
        </div>
        <textarea rows="2" ${v.builtin ? 'readonly' : ''} placeholder="The exact words spoken in the clip">${escapeHtml(v.transcript)}</textarea>
        ${v.builtin ? '<div class="a8-hint">Built-in voices are read-only.</div>' : `
        <div class="a8-row">
            <button type="button" data-act="stt">Transcribe</button>
            <button type="button" data-act="save" disabled>Save transcript</button>
            <span class="a8-spacer"></span>
            <button type="button" data-act="del" class="danger">Delete</button>
        </div>`}`;
    const ta = el.querySelector('textarea');
    const btn = act => el.querySelector(`[data-act="${act}"]`);
    ta.addEventListener('input', () => { if (btn('save')) btn('save').disabled = ta.value.trim() === v.transcript; });

    btn('try').addEventListener('click', async () => {
        const b = btn('try');
        b.disabled = true; b.dataset.busy = '1'; b.textContent = 'Speaking...';
        try {
            const blob = await api(vurl(v.name, '/try'), json('POST', {}));
            const a = new Audio(URL.createObjectURL(blob));
            a.addEventListener('ended', () => URL.revokeObjectURL(a.src));
            await a.play();
        } catch (e) { flash(errWords(e), 'error'); }
        delete b.dataset.busy; b.disabled = !ready; b.textContent = 'Try it';
    });

    btn('stt')?.addEventListener('click', async () => {
        const b = btn('stt');
        const run = async (cloud_ok) => {
            b.disabled = true; b.textContent = 'Listening...';
            try {
                const r = await api(vurl(v.name, '/transcribe'), json('POST', { cloud_ok }));
                ta.value = r.transcript;
                btn('save').disabled = ta.value.trim() === v.transcript;
            } catch (e) {
                if (e.status === 409 && !cloud_ok) {
                    showConfirm(`${e.message} Send this clip there to transcribe it?`, () => run(true));
                } else flash(errWords(e), 'error');
            }
            b.disabled = false; b.textContent = 'Transcribe';
        };
        run(false);
    });

    btn('save')?.addEventListener('click', async () => {
        try {
            const r = await api(vurl(v.name), json('PUT', { transcript: ta.value }));
            Object.assign(v, r.voice);
            flash(`Saved ${v.name}'s transcript`, 'success');
            el.replaceWith(card(v, box));
            fillVoiceSelect(box);
        } catch (e) { flash(errWords(e), 'error'); }
    });

    btn('del')?.addEventListener('click', () => {
        showConfirm(`Delete the voice "${v.name}"? Its clip and transcript are removed.`, async () => {
            try {
                await api(vurl(v.name), json('DELETE'));
                lib.voices = lib.voices.filter(x => x.name !== v.name);
                el.remove();
                fillVoiceSelect(box);
            } catch (e) { flash(errWords(e), 'error'); }
        });
    });
    return el;
}

function renderLibrary(box) {
    box.querySelector('#a8-stt').innerHTML = sttLine();
    const list = box.querySelector('#a8-list');
    list.innerHTML = '';
    lib.voices.forEach(v => list.appendChild(card(v, box)));
    fillVoiceSelect(box);
    applyReady(box);
}

async function loadLibrary(box) {
    try {
        lib = await api(API, {}, 20000);
    } catch (e) {
        box.querySelector('#a8-stt').textContent = errWords(e, "Couldn't load the voices.");
        return;
    }
    renderLibrary(box);
}

// Clip helpers: the browser decodes the file (any format it can play), so a
// long clip is trimmed HERE before upload, cut at the quietest moment near
// each edge rather than mid-word. The server still converts and checks.
const BEST = 15, MAX = 30, MIN = 3;

function mono(buf) {
    const out = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) {
        const d = buf.getChannelData(c);
        for (let i = 0; i < d.length; i++) out[i] += d[i] / buf.numberOfChannels;
    }
    return out;
}

function quietest(pcm, sr, t, win = 0.6) {
    const frame = Math.round(0.03 * sr);
    let best = t, low = Infinity;
    for (let at = Math.max(0, t - win); at <= Math.min(pcm.length / sr, t + win); at += 0.015) {
        const i0 = Math.round(at * sr) - frame / 2;
        let e = 0;
        for (let i = Math.max(0, i0); i < Math.min(pcm.length, i0 + frame); i++) e += pcm[i] * pcm[i];
        if (e < low) { low = e; best = at; }
    }
    return best;
}

function wavBlob(pcm, sr) {
    const v = new DataView(new ArrayBuffer(44 + pcm.length * 2));
    const str = (o, t) => [...t].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
    str(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); str(8, 'WAVEfmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
    return new Blob([v], { type: 'audio/wav' });
}

function wireUpload(box) {
    const $ = id => box.querySelector(id);
    const file = $('#a8-file'), name = $('#a8-name'), go = $('#a8-upload');
    const note = $('#a8-note'), info = $('#a8-info'), trim = $('#a8-trim');
    const tStart = $('#a8-t-start'), tLen = $('#a8-t-len'), tPlay = $('#a8-t-play');
    let clip = null;        // {pcm, sr, dur} when the browser could read the file
    let ctx = null, playing = null;

    const say = (el, text, err = false) => { el.textContent = text; el.classList.toggle('err', err); };

    const window_ = () => {
        let s = Math.max(0, Math.min(parseFloat(tStart.value) || 0, clip.dur - MIN));
        let e = Math.min(clip.dur, s + Math.max(MIN, Math.min(MAX, parseFloat(tLen.value) || BEST)));
        if (s > 0) s = quietest(clip.pcm, clip.sr, s);
        if (e < clip.dur) e = quietest(clip.pcm, clip.sr, e);
        if (e - s < MIN) e = Math.min(clip.dur, s + MIN);
        return [s, e];
    };

    file.addEventListener('change', async () => {
        const f = file.files[0];
        clip = null; trim.hidden = true; say(note, ''); say(info, '');
        if (!f) return;
        if (!name.value.trim()) {
            name.value = f.name.replace(/\.[^.]+$/, '').toLowerCase().replace(/[^a-z0-9_-]+/g, '_').slice(0, 40);
        }
        say(info, 'Reading the clip...');
        try {
            const ac = new AudioContext();
            const buf = await ac.decodeAudioData(await f.arrayBuffer());
            ac.close();
            clip = { pcm: mono(buf), sr: buf.sampleRate, dur: buf.duration };
        } catch {
            say(info, "Your browser can't read this file, so Sapphire will try it as is (WAV, MP3, FLAC or OGG).");
            return;
        }
        const d = clip.dur.toFixed(1);
        if (clip.dur < MIN) {
            say(info, `This clip is ${d}s. Audio8 needs at least ${MIN}s of speech (5-15s is best). Pick a longer clip.`, true);
        } else if (clip.dur > MAX) {
            tStart.value = 0; tLen.value = BEST; trim.hidden = false;
            say(info, `This clip is ${d}s, and Audio8 takes ${MAX}s at most (5-15s is best). Sapphire will keep only the part below, cut at the nearest pause. Set where your cleanest speech starts and press Preview to hear it.`);
        } else if (clip.dur > BEST) {
            tStart.value = 0; tLen.value = Math.round(clip.dur * 2) / 2; trim.hidden = false;
            say(info, `This clip is ${d}s. That works, but 5-15s clips sound steadier. Shorten it below if you like.`);
        } else {
            say(info, `This clip is ${d}s. Good length.`);
        }
    });

    tPlay.addEventListener('click', () => {
        if (!clip) return;
        try { playing?.stop(); } catch {}
        ctx = ctx || new AudioContext();
        const [s, e] = window_();
        const b = ctx.createBuffer(1, clip.pcm.length, clip.sr);
        b.copyToChannel(clip.pcm, 0);
        playing = ctx.createBufferSource();
        playing.buffer = b;
        playing.connect(ctx.destination);
        playing.start(0, s, e - s);
        say(note, `Previewing ${s.toFixed(1)}s to ${e.toFixed(1)}s (${(e - s).toFixed(1)}s).`);
    });

    go.addEventListener('click', async () => {
        const f = file.files[0];
        if (!f) { flash('Pick an audio file first', 'warning'); return; }
        if (clip && clip.dur < MIN) { flash(info.textContent, 'error'); return; }
        try { playing?.stop(); } catch {}
        const fd = new FormData();
        if (clip) {
            const [s, e] = clip.dur > BEST ? window_() : [0, clip.dur];
            fd.append('file', wavBlob(clip.pcm.subarray(Math.round(s * clip.sr), Math.round(e * clip.sr)), clip.sr), 'clip.wav');
        } else {
            fd.append('file', f);
        }
        fd.append('name', name.value.trim());
        go.disabled = true; go.textContent = 'Adding...';
        say(note, '');
        try {
            const r = await api(API, { method: 'POST', body: fd }, 180000);
            lib.voices = [...lib.voices.filter(x => x.name !== r.voice.name), r.voice]
                .sort((a, b) => (a.builtin - b.builtin) || a.name.localeCompare(b.name));
            renderLibrary(box);
            say(note, r.note || `Added ${r.voice.name} (${r.voice.seconds}s). Check its transcript below, then pick it in the Sampler.`);
            flash(`Added voice ${r.voice.name}`, 'success');
            file.value = ''; name.value = ''; clip = null; trim.hidden = true; say(info, '');
        } catch (e) {
            say(note, errWords(e), true);
            flash(errWords(e), 'error');
        }
        go.disabled = false; go.textContent = 'Add voice';
    });
}

export default {
    name: 'audio8',

    init() {
        injectStyles();
        registerPluginSettings({
            id: 'audio8',
            name: 'Audio8',
            icon: '🎙️',
            helpText: 'Zero-shot voice cloning on your GPU. A voice is a short clip (5-15s of clean speech) plus the exact words spoken in it.',

            render(container, settings) {
                container.innerHTML = `
                    <div class="a8">
                        <section class="a8-card a8-status" id="a8-status">
                            <div class="a8-name">Checking Audio8...</div>
                            <div class="a8-hint"></div>
                            <progress hidden></progress>
                        </section>
                        <section class="a8-card">
                            <h4>Sampler</h4>
                            <div class="a8-row">
                                <label for="a8-s-voice">Voice</label> <select id="a8-s-voice"></select>
                                <label for="a8-s-speed">Speed</label> <input type="number" id="a8-s-speed" value="1" min="0.75" max="1.6" step="0.05" style="width:70px">
                                <label for="a8-s-pitch">Pitch</label> <input type="number" id="a8-s-pitch" value="1" min="0.5" max="2" step="0.02" style="width:70px">
                            </div>
                            <textarea id="a8-s-text" rows="4" maxlength="2000" placeholder="Type anything for the voice to say. Ctrl+Enter speaks."></textarea>
                            <div class="a8-row">
                                <button type="button" id="a8-s-go">Speak</button>
                                <button type="button" id="a8-s-dl" disabled>Download MP3</button>
                                <span class="a8-spacer"></span>
                                <span class="a8-hint" id="a8-s-count">0 / 2000</span>
                            </div>
                            <audio id="a8-s-audio" controls hidden></audio>
                            <div class="a8-hint">Every take is fresh: Speak again for another reading. Only share voices you have the right to use.</div>
                        </section>
                        <section id="a8-general"></section>
                        <section>
                            <h4>Voices</h4>
                            <div class="a8-hint" id="a8-stt">Loading voices...</div>
                            <div class="a8-card" style="margin-top:var(--space-sm)">
                                <div class="a8-name">Add a voice</div>
                                <div class="a8-row">
                                    <input type="file" id="a8-file" accept="audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.opus,.webm">
                                    <input type="text" id="a8-name" placeholder="name (letters, numbers, - _)" maxlength="40">
                                    <button type="button" id="a8-upload">Add voice</button>
                                </div>
                                <div class="a8-hint" id="a8-info"></div>
                                <div class="a8-row" id="a8-trim" hidden>
                                    <label for="a8-t-start">Start at</label> <input type="number" id="a8-t-start" min="0" step="0.5" value="0" style="width:70px"> s
                                    <label for="a8-t-len">Length</label> <input type="number" id="a8-t-len" min="3" max="30" step="0.5" value="15" style="width:70px"> s
                                    <button type="button" id="a8-t-play">Preview</button>
                                </div>
                                <div class="a8-hint">Most audio files work: 3-30s (5-15s is best), one speaker, no music. A longer clip can be trimmed here. Saved on this computer in user/audio8-voices/. Only add a voice you have permission to clone.</div>
                                <div class="a8-hint" id="a8-note"></div>
                            </div>
                            <div id="a8-list" style="display:flex;flex-direction:column;gap:var(--space-sm);margin-top:var(--space-sm)"></div>
                        </section>
                    </div>`;
                renderGeneral(container, settings || {});
                wireUpload(container);
                wireSampler(container);
                loadLibrary(container);
                watchStatus(container);
            },

            load: () => pluginsAPI.getSettings('audio8'),
            save: (s) => pluginsAPI.saveSettings('audio8', s),

            getSettings(container) {
                const num = (id, d) => { const n = parseFloat(container.querySelector(id).value); return Number.isFinite(n) ? n : d; };
                return {
                    voice: container.querySelector('#a8-voice').value || 'sapphire',
                    temperature: num('#a8-temp', 0.8),
                    port: Math.round(num('#a8-port', 5014)),
                    pieces: container.querySelector('#a8-pieces').checked,
                };
            },
        });
    },

    destroy() {},
};
