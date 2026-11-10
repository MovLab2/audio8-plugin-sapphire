"""Fast decode loop for Audio8 (batch 1): static buffers, one CUDA graph per
frame, optionally fused by torch.compile.

Each frame decodes as the model's own generate() does: draw the semantic
token from the last slow-step logits (with RAS), write the 10 codebooks through
the fast layers, then one slow step on the new column. Filtering, temperature
and the RAS window follow the model exactly, in the model's own dtype. What
changes is where tensors live: KV caches, the attention mask and every per-step
index are persistent buffers, so the whole frame replays as one graph instead
of ~3,000 separate kernel launches (2026-11-08: 23 ms -> 4.3 ms per frame on a
5090, 2.0x -> 10.9x realtime).

The loop is modelled on the model's generate() and _generate_codebooks()
(audio8_tts by Edge0, Apache License 2.0; see NOTICE) closely enough to decode
identically, restructured around persistent buffers. The model's modules
(layers, embeddings, _slow_step, _prepare_prompt) are called, not copied. The
sampling is written independently from the published methods: top-k and
nucleus filtering (Holtzman et al., 2019), Gumbel-max sampling, and
repetition-aware sampling (RAS, VALL-E 2, Chen et al., 2024).

One deliberate difference: a draw is Gumbel-max on fp32 noise, an exact sample
of softmax(scores). The model's own sampler uses noise in the model's dtype,
which under bf16 lands about 1% (total variation) off softmax. Greedy decoding
is unaffected.

tools/parity.py checks this against the model: greedy decoding identical to
model.generate under deterministic fp32, the filter keeping the same tokens as
the model's own in fp32 and bf16, the RAS window, and the sampler matching
softmax. It reads the model's private methods, so re-run it after a model
revision bump.
"""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


class FastGen:
    def __init__(self, model, use_graph=True):
        self.m = m = model
        c = m.config
        d = self.dev = m.device
        dt = next(m.parameters()).dtype     # filter math runs in this, as the model's does
        self.nc, self.L, self.eos = c.num_codebooks, c.max_seq_len, c.eos_token_id
        # One set of KV caches for the life of the process: the graph holds
        # their addresses. generate() would replace them, so pin the setup.
        m._setup_generation_caches(1, self.L, dt)
        m._setup_generation_caches = lambda *a, **k: None
        # The only tokens a semantic step may emit, in vocab order (EOS sits
        # below the semantic range), so a greedy tie breaks as the model's does.
        self.cand = torch.cat((torch.tensor([c.eos_token_id], device=d),
                               torch.arange(c.semantic_begin_id, c.semantic_end_id + 1, device=d)))
        self.col = torch.zeros((1, self.nc + 1, 1), dtype=torch.long, device=d)
        self.pos = torch.zeros((1,), dtype=torch.long, device=d)
        self.mask = torch.zeros((1, self.L), dtype=torch.long, device=d)
        self.logits = torch.zeros((1, c.vocab_size), dtype=dt, device=d)
        self.hidden = None
        self.recent = torch.full((1, c.ras_window_size), -1, dtype=torch.long, device=d)
        self.first = torch.ones((), dtype=torch.bool, device=d)
        # Thresholds and temperatures in the model's dtype, like the model's own
        # (bf16 0.95 is 0.9492); a fp32 or Python scalar would compare unrounded.
        self.temp = torch.tensor(0.8, dtype=dt, device=d)
        self.top_p = torch.tensor(0.95, dtype=dt, device=d)
        self.ras_temp = torch.tensor(c.ras_temperature, dtype=dt, device=d).clamp_min(1e-5)
        self.ras_top_p = torch.tensor(c.ras_top_p, dtype=dt, device=d)
        self.top_k = 50                     # a graph constant; the server always asks 50
        self.out_sem = torch.zeros((1,), dtype=torch.long, device=d)
        self.out_codes = torch.zeros((1, self.nc), dtype=torch.long, device=d)
        self.fpos = [torch.tensor([p], device=d) for p in range(self.nc)]
        ones = torch.ones((1, self.nc), dtype=torch.bool, device=d)
        self.fmask = [m._causal_mask(ones, self.fpos[p], self.nc) for p in range(self.nc)]
        self.frope = [m.fast_freqs_cis[self.fpos[p]] for p in range(self.nc)]
        self.use_graph = use_graph
        self.graph = None
        self.sample = True

    def _filter(self, scores, top_p):
        """Top-k plus nucleus, the variant this model decodes with: of the
        top_k best tokens, keep each while the probability mass up to and
        including it is <= top_p (the best token always stays). The mass is
        measured against the softmax over ALL scores, in the scores' own dtype
        (top_p a tensor of that dtype too): bf16 rounding moves which tail token
        sits at the cutoff, and the model was tuned with its own precision.
        Others become -inf. The running mass is summed over the whole row in
        descending order, then cut to top_k: on CUDA a bf16 cumsum over only
        top_k elements rounds differently and moves ~2% of cutoffs."""
        best, where = torch.topk(scores, self.top_k, dim=-1)
        probs = torch.sort(torch.softmax(scores, dim=-1), dim=-1, descending=True).values
        mass = probs.cumsum(dim=-1)[..., :self.top_k]
        keep = mass <= top_p
        keep[..., 0] = True
        out = torch.full_like(scores, float("-inf"))
        return out.scatter(-1, where, best.masked_fill(~keep, float("-inf")))

    @staticmethod
    def _gumbel(logits):
        """Gumbel-max: argmax(logits + G) with G = -log(-log U) is a draw from
        softmax(logits). U is fp32 and kept off 0 and 1 so G stays finite."""
        u = torch.rand(logits.shape, device=logits.device).clamp_(1e-10, 1.0 - 1e-7)
        return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)

    def _scaled(self, scores, top_p, temp):
        return self._filter(scores, top_p) / temp

    def _semantic(self):
        """Next semantic token (or EOS) from the last slow step. Filtering only
        the legal candidates is exact: every other logit is masked anyway."""
        sc = self.logits[:, self.cand]
        regular = self._scaled(sc, self.top_p, self.temp.clamp_min(1e-5))
        if not self.sample:
            return self.cand[regular.argmax(dim=-1)]
        token = self.cand[self._gumbel(regular)]
        # RAS: a semantic token already in the recent window is replaced by a
        # draw from the wider fallback distribution. Both draws are made every
        # frame, so the graph has no branch.
        wide = self._scaled(sc, self.ras_top_p, self.ras_temp)
        fallback = self.cand[self._gumbel(wide)]
        repeat = (self.recent == token[:, None]).any(dim=1) & (token != self.eos)
        return torch.where(repeat, fallback, token)

    def _fast(self, hidden, p):
        m = self.m
        for layer in m.fast_layers:
            hidden = layer(hidden, self.frope[p], self.fmask[p], self.fpos[p])
        return m.fast_output(m.fast_norm(hidden))[:, -1]

    def _codebooks(self, semantic):
        """One frame's codebooks from the fast transformer. Slot 0 takes the
        slow model's hidden state (projected in); its output is unused, the
        step only fills the KV cache. Codebook 0 is the semantic token moved
        into codebook range; codebook p is read from slot p, which is fed the
        embedding of codebook p-1."""
        m, c = self.m, self.m.config
        self._fast(m.fast_project_in(self.hidden), 0)
        code = (semantic - c.semantic_begin_id).clamp(0, c.codebook_size - 1)
        codes = [code]
        for p in range(1, self.nc):
            scores = self._scaled(self._fast(m.fast_embeddings(code)[:, None], p),
                                  self.top_p, self.temp.clamp_min(1e-5))
            code = self._gumbel(scores) if self.sample else scores.argmax(dim=-1)
            codes.append(code)
        return torch.stack(codes, dim=1)

    def _frame(self):
        sem = self._semantic()
        codes = self._codebooks(sem)
        self.out_sem.copy_(sem)
        self.out_codes.copy_(codes)
        # The model's window skips the first frame's token, then takes each one.
        push = torch.where(self.first, -1, sem)
        self.recent.copy_(torch.cat((self.recent[:, 1:], push[:, None]), dim=1))
        self.first.fill_(False)
        self.col.copy_(torch.cat((sem[:, None], codes), dim=1).unsqueeze(-1))
        self.mask.scatter_(1, self.pos[None], 1)
        with sdpa_kernel(SDPBackend.MATH):
            lg, hd = self.m._slow_step(self.col, self.pos, self.pos[None], self.mask)
        self.logits.copy_(lg)
        self.hidden.copy_(hd)
        self.pos.add_(1)

    def _capture(self):
        state = (self.col, self.pos, self.mask, self.logits, self.hidden, self.recent, self.first)
        snap = [t.clone() for t in state]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._frame()                     # compile + allocator warm-up
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._frame()
        for t, v in zip(state, snap):
            t.copy_(v)
        self.graph = graph

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, do_sample=True):
        """Codes [1, num_codebooks, T] for processor `inputs` (batch 1).
        The graph is captured sampling; greedy (parity checks) runs eager."""
        frames = list(self.stream(inputs, max_new_tokens, temperature, top_p, do_sample))
        if not frames:
            return torch.empty((1, self.nc, 0), dtype=torch.long, device=self.dev)
        return torch.stack(frames, dim=2)

    @torch.inference_mode()
    def stream(self, inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, do_sample=True):
        """Each frame's codes [1, num_codebooks] as it decodes, until EOS."""
        if self.use_graph and not do_sample:
            raise ValueError("the graph is captured for sampling; greedy runs eager")
        m = self.m
        self.sample = do_sample
        self.temp.fill_(temperature)
        self.top_p.fill_(top_p)
        prompt, pmask = m._prepare_prompt(**inputs)
        W = prompt.shape[-1]
        if W >= self.L:
            raise ValueError(f"prompt length {W} must be smaller than {self.L}")
        lg, hd = m._slow_step(prompt, torch.arange(W, device=self.dev),
                              pmask.cumsum(-1).sub(1).clamp_min(0), pmask)
        if self.hidden is None:
            self.hidden = torch.zeros_like(hd)
        self.logits.copy_(lg)
        self.hidden.copy_(hd)
        self.mask.zero_()
        self.mask[:, :W] = pmask
        self.pos.fill_(W)
        self.recent.fill_(-1)
        self.first.fill_(True)
        if self.use_graph and self.graph is None:
            self._capture()
        for _ in range(min(max_new_tokens, self.L - W)):
            if self.use_graph:
                self.graph.replay()
            else:
                self._frame()
            if int(self.out_sem.item()) == self.eos:
                return
            yield self.out_codes.clone()
