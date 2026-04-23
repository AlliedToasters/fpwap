# fpwap roadmap

Narrative context only. **Actionable work lives in GitHub issues**
(<https://github.com/AlliedToasters/fpwap/issues>); this doc is the
motivation and point-in-time snapshot behind them.

---

## Current state (2026-04-23)

Hero milestone banked (see `memory/hero_model_milestone.md`):

- Llama-3.3-70B streaming @ 10k × 128 on RTX 5090: **1,221 tok/s** (29% above SPEC ≥950)
- Llama-3.1-8B preloaded: 11.9k tok/s (2.4× SPEC ≥5k)
- Bit-exact vs naive HF forward on Llama-3.2-1B at all 16 layers
- Cross-family (GPT-2 + Llama/Mistral/Qwen2/Gemma via structural matcher)
- All four hooks + sub-layer WriteBack, four reference callbacks shipped
- ProfileReport, MemmapBackend, pinned CPU buffer + async D2H, worker-thread weight prefetch
- 70 non-GPU + 4 GPU tests green; mypy + ruff clean
- `scripts/benchmark.py --mode naive` reproduces SPEC §17 baseline; 8B @ 1024×128 shows 7.25× ratio

The engine + callback surface is adoption-ready in isolation. The
remaining work is **integration ergonomics + a benchmark that
demonstrates the amortization thesis**, not missing engine features.

---

## Why these issues exist — lmprobe integration review (2026-04-23)

The issues open against this repo are shaped by a read of the lmprobe
team's runnable harness at
`../deception_detection/scripts/bench/benchmark_offload_backends.py`
(head-to-head over 7 offload backends: accelerate-gpu,
accelerate-cpu-static, accelerate-cpu-stream, accelerate-disk-stream,
accelerate-shard-disk, lmprobe-chunked, lmprobe-disk; `--iters` for
cold/warm; uses `torch.cuda.max_memory_allocated` for peak VRAM;
fingerprint = per-batch mean of non-pad residuals, cosine-sim'd for
correctness).

Existing numbers on that harness:

- 70B: accelerate-cpu fastest per-batch; accelerate-disk amortizes long jobs (~214 s one-time weight copy, then ~37 s/iter steady state)
- 405B: accelerate-mmap is the only stock-accelerate path that loads; lmprobe-disk also works; mmap uses 3× less VRAM
- 8B: accelerate-cpu-stream vs lmprobe-chunked → 3.5× faster, 6.6× less peak VRAM
- 70B: accelerate-disk-stream vs lmprobe-disk → 2.6× faster per-batch, 7× less peak VRAM

**Core critique:** the harness times single-batch extraction. fpwap's
headline claim (weight-I/O `O(N_layers)` not `O(N_batches × N_layers)`)
only shows up at dataset scale. `--iters` measures steady-state, not
amortization. Fpwap needs a different shape of measurement — the
N-scaling benchmark in #4 is the answer.

**Narrative framing:** don't lead with 8B (dense-fits-VRAM, where fpwap
can only tie accelerate-gpu). Lead with 70B on 32 GB VRAM — the regime
that justifies fpwap's existence. The README Status table currently
mis-leads here; #5 fixes it.

---

## Issue map

Integration ergonomics (what adopters need to not rewrite our plumbing):

- #6 — auto-resolve HF model-id to snapshot inside `Sweep(...)` (small)
- #7 — `run_fpwap` adapter drafted against the lmprobe harness contract
- #8 — `Extractor` handle for multi-sweep reuse of empty_model + accel_index

Benchmarks that move the adoption needle:

- #4 — **the killer plot**: 70B + 8B N-scaling, dataset-scale wall-clock and peak VRAM
- #9 — multi-dataset warm-start overhead (gates on #8)
- #11 — 70B bit-exactness (blocked on a host with ≥141 GB RAM)

Narrative cleanup:

- #5 — README Status leads with 70B-on-32GB, not 8B ratio

Deferred / blocked but tracked:

- #10 — variable-length batches (architectural; waiting for pad-waste to be a real bottleneck)
- #11 — 70B bit-exactness (hardware-limited on this box)

Out-of-scope for fpwap-solo work:

- lmprobe-side AUROC-parity test harness — lives in the consumer repo;
  coordinate with integration owner when relevant. No fpwap issue.

Scope-declined (no concrete use case):

- Checkpoint/resume — 18-min hero runs don't justify it.
- NVMe-backed `ResidualBuffer` — pinned-CPU buffer handles 10k × 128 @ 70B within 128 GB RAM.
- Streaming-path `verify=True` — inherent limitation (naive ref needs full model resident); covered by preloaded verify.

---

## Notes for next operator

- `scripts/benchmark.py --mode naive` requires the model to fit in CPU
  RAM for `accelerate.cpu_offload`. 70B on 128 GB RAM will OOM; use an
  `accelerate-disk-stream` equivalent (not wired here) for 70B naive.
  Relevant to #11.
- `torch.cuda.reset_peak_memory_stats()` is called per-N in
  `scripts/bench_n_scaling.py` so each row's `peak_gpu_mb` is
  independent. `ru_maxrss` is process-lifetime so `peak_cpu_mb` is
  monotonic — interpret as "RAM high-water so far."
- The existing integration recipe memory (`integration_recipe.md`) —
  `from_fpwap()` classmethod at the activation-source dispatch layer —
  is still the shape #8 (`Extractor` handle) should consume cleanly.
