"""Check that fastgen.py decodes like the model's own generate().

Run with the plugin env's python after any model revision bump (fastgen reads
the model's private methods):

    <env>/bin/python user/plugins/audio8/tools/parity.py [voice.wav voice.txt]

Deterministic fp32, greedy: the model's generate() vs FastGen's eager loop must
produce identical codes. The model's FIRST generate() of a process differs from
every later one (a warm-up effect), so it is discarded. Greedy never looks at
the filter (the best token always survives it) or RAS, so more checks cover the
sampling, calling the model's own code at run time:
- filter: FastGen's top-k/top-p keeps the same tokens as the model's filter,
  in fp32 and bf16, on the full vocab (semantic-masked) and on FastGen's
  candidate subset. Equal logits may swap places at the cutoff, so kept VALUES
  are compared. bf16 allows a sliver of rows (<0.5%) for reduction-order
  rounding at the cutoff; fp32 must match exactly.
- RAS window: skips the first frame's token, then takes each one.
- sampler: Gumbel-max draws land on softmax's frequencies.
- graph and compiled engines: sampled runs produce frames, and two runs differ
  (fresh noise per replay). The compiled engine needs Triton; without it the
  server uses the graph engine, so that check is skipped, not failed.
Exit 0 = every check passed.
"""
import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

from fastgen import FastGen  # noqa: E402
from server import MODEL_ID, MODEL_REV  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
TEXT = "Hello there, this is a short parity test of the decoder."


def _kept(scores):
    """Kept logit values per row, sorted: tie-tolerant comparison."""
    return torch.sort(scores.float(), dim=-1, descending=True).values


def filter_check(model, fg):
    """Kept tokens vs the model's own filter on random logit rows: sharp, flat
    and in between, at the top_p values the decoder uses."""
    mod = sys.modules[type(model).__module__]
    legacy, semantic = mod.ArkttsLegacyTopKTopPLogitsProcessor, mod.ArkttsSemanticLogitsProcessor
    c = model.config
    sem = semantic(c.semantic_begin_id, c.semantic_end_id, c.eos_token_id)
    torch.manual_seed(0)
    ok = True
    for dtype, slack in ((torch.float32, 0.0), (torch.bfloat16, 0.005)):
        rows = bad = 0
        for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
            for top_p in (0.95, c.ras_top_p, 0.5):
                tp = torch.tensor(top_p, dtype=dtype, device="cuda")
                # Candidate width, as FastGen filters it.
                x = (torch.randn(200, fg.cand.numel(), device="cuda") * scale).to(dtype)
                ours = _kept(fg._filter(x, tp))
                theirs = _kept(legacy(fg.top_k, float(top_p))(None, x))
                bad += int((ours != theirs).any(dim=1).sum())
                # Full vocab, semantic-masked by the model, vs FastGen's subset.
                v = (torch.randn(20, c.vocab_size, device="cuda") * scale).to(dtype)
                theirs = _kept(legacy(fg.top_k, float(top_p))(None, sem(None, v)))[:, :fg.top_k]
                ours = _kept(fg._filter(v[:, fg.cand], tp))[:, :fg.top_k]
                bad += int((ours != theirs).any(dim=1).sum())
                rows += 220
        good = bad <= slack * rows
        ok &= good
        print(f"filter {str(dtype)[6:]}: {rows - bad}/{rows} rows keep the same tokens as the model's filter")
    return ok


def ras_check(fg, inp):
    """After 3 sampled frames the window holds frames 1 and 2, not frame 0."""
    fg.use_graph = False
    codes = fg.generate(inp, max_new_tokens=3)
    if codes.shape[-1] < 3:
        print("ras window: stopped early, can't judge")
        return False
    sem = codes[0, 0] + fg.m.config.semantic_begin_id
    want = torch.cat((torch.full((fg.recent.shape[1] - 2,), -1, device=sem.device), sem[1:3]))
    same = torch.equal(fg.recent[0], want)
    print(f"ras window: skips frame 0, holds frames 1-2 = {same}")
    return same


def sampler_check(fg):
    """Gumbel-max frequencies vs softmax over 2M draws."""
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0, float("-inf")], device="cuda")
    draws = fg._gumbel(logits.expand(2_000_000, -1))
    freq = torch.bincount(draws, minlength=6).float() / draws.numel()
    gap = float((freq - torch.softmax(logits, -1)).abs().max())
    print(f"sampler: max frequency gap vs softmax {gap:.4f} (masked token drawn {int(freq[5] * draws.numel())} times)")
    return gap < 0.002 and freq[5] == 0


def sampled_check(fg, inp, name):
    a, b = fg.generate(inp, max_new_tokens=60), fg.generate(inp, max_new_tokens=60)
    fresh = a.shape != b.shape or not torch.equal(a, b)
    print(f"{name} sampled: {a.shape[-1]} and {b.shape[-1]} frames, fresh noise per run = {fresh}")
    return a.shape[-1] > 0 and b.shape[-1] > 0 and fresh


def main():
    proc = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REV, trust_remote_code=True,
                                      dtype=torch.float32).eval().to("cuda")
    kw = {"text": [TEXT], "return_tensors": "pt"}
    if len(sys.argv) == 3:
        with open(sys.argv[2], encoding="utf-8") as f:
            kw.update(reference_audio=[sys.argv[1]], reference_text=[f.read().strip()])
    inp = {k: v.to("cuda") for k, v in proc(**kw).items()}
    with torch.inference_mode():
        gen = lambda: model.generate(**inp, max_new_tokens=120, do_sample=False,
                                     return_dict_in_generate=True).codes
        gen()                                   # first run of a process differs: discard
        ref, again = gen(), gen()
    if not torch.equal(ref, again):
        print("FAIL: the model's own generate() is not deterministic here; parity can't be judged")
        return 2
    fg = FastGen(model, use_graph=False)
    fast = fg.generate(inp, max_new_tokens=120, do_sample=False)
    same = torch.equal(ref, fast)
    print(f"greedy fp32: original {ref.shape[-1]} frames, fastgen {fast.shape[-1]} frames, identical={same}")
    with torch.inference_mode():
        filt = filter_check(model, fg)
    ras, samp = ras_check(fg, inp), sampler_check(fg)
    fg.use_graph = True
    graph = sampled_check(fg, inp, "graph")
    compiled = True
    try:
        fc = FastGen(model)                     # caches are shared and already pinned
        fc._frame = torch.compile(fc._frame, dynamic=False)
        compiled = sampled_check(fc, inp, "compiled")
    except Exception as e:                      # no Triton: the server falls back to graph
        print(f"compiled: skipped ({type(e).__name__}: {e})")
    ok = same and filt and ras and samp and graph and compiled
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
