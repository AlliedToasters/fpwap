# fpwap roadmap

Living action plan. Captures the adoption-review blob dropped 2026-04-23
from the lmprobe/deception_detection integration side, plus ranked
priorities and what's in-flight vs. deferred.

Update this doc as priorities shift. Don't let items fall off the bottom
of a conversation.

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

The engine + callback surface is adoption-ready in isolation. The gap to
real-use-in-lmprobe is **integration ergonomics + a benchmark that
demonstrates the amortization thesis**, not missing engine features.

---

## Incoming feedback — lmprobe integration review (2026-04-23)

From a read of the runnable harness at
`../deception_detection/scripts/bench/benchmark_offload_backends.py`
(head-to-head over 7 offload backends: accelerate-gpu, accelerate-cpu-static,
accelerate-cpu-stream, accelerate-disk-stream, accelerate-shard-disk,
lmprobe-chunked, lmprobe-disk; `--iters` for cold/warm; uses
`torch.cuda.max_memory_allocated` for peak VRAM; fingerprint = per-batch
mean of non-pad residuals, cosine-sim'd for correctness).

Existing numbers on that harness:
- 70B: accelerate-cpu fastest per-batch; accelerate-disk amortizes long jobs (~214 s one-time weight copy, then ~37 s/iter steady state)
- 405B: accelerate-mmap is the only stock-accelerate path that loads; lmprobe-disk also works; mmap uses 3× less VRAM
- 8B: accelerate-cpu-stream vs lmprobe-chunked → 3.5× faster, 6.6× less peak VRAM
- 70B: accelerate-disk-stream vs lmprobe-disk → 2.6× faster per-batch, 7× less peak VRAM

**Core critique:** the harness times single-batch extraction. fpwap's
headline claim (weight-I/O `O(N_layers)` not `O(N_batches × N_layers)`)
only shows up at dataset scale. `--iters` measures steady-state, not
amortization. Fpwap needs a different shape of measurement.

**Narrative suggestion:** don't lead with 8B (dense-fits-VRAM, where fpwap
can only tie accelerate-gpu). Lead with 70B on 32 GB VRAM — the regime
that justifies fpwap's existence.

### Integration gaps (team flagged these)

1. **Concrete fpwap-stream row in the benchmark harness.** Needs
   `run_fpwap(model_id, pretokenized_batch, hf_layers)` returning the
   same `(load_s, extract_times, fingerprint, peak_VRAM)` contract as
   the other backends. Matching the harness means zero argument from
   reviewers.
2. **Multi-sweep reuse of empty-weights model + accel_index.** The
   experiment pass is 15+ eval datasets per run; `Sweep(model="meta-llama/…")`
   rebuilds `build_empty_model_and_index` every time. For 70B that's
   real wall-clock burned. Expose an `Extractor`-like handle that owns
   `(empty_model, accel_index, snapshot_dir)` across sweeps, or document
   that users should hand in a pre-built `nn.Module` + their own streamer.
3. **Variable-length input batches.** fpwap requires single `seq_len`;
   their datasets have very different length distributions (alpaca ~50,
   ai_liar 400+). Forcing pad-to-max is compute wasted. Not a blocker;
   worth flagging.
4. **Model-id path that resolves HF cache automatically.**
   `Sweep(model="meta-llama/…")` with `_OffloadStreamer` currently
   expects a snapshot dir. Integration wants to pass the HF model id
   and have fpwap call `snapshot_download(…, local_files_only=True)`
   internally.
5. **AUROC-parity test harness.** Correctness bar isn't cosine-sim of
   residuals, it's "the linear probe trained on fpwap activations
   reproduces the paper's AUROCs." A tiny integration test that runs
   the full extract → probe → eval on 8B + one small dataset would
   catch silent drift that bit-exactness tests don't.

### Benchmarks that would actually move the needle (ranked)

1. **Dataset-scale head-to-head on 70B.** Extract last-token
   `residual_post` @ layer 22 over N prompts, seq_len 256, sweeping
   N ∈ {8, 64, 512, 4096}. Plot wall-clock vs N for each backend. This
   is **the killer plot**. fpwap's curve should be near-flat slope;
   accelerate-cpu/disk are linear in N. The crossover point is the
   whole story. Any N below crossover, use accelerate; above, use fpwap.
2. **End-to-end wall-clock for one full experiment pass.** Reproduce the
   paper's 70B eval (15 datasets, ~1k prompts each) with each backend.
   Total wall-clock, not per-batch. accelerate-cpu is the number to
   beat — currently their "winner" — and this is where multi-dataset
   amortization (gap #2) shows up or doesn't.
3. **Bit-exactness at 70B against accelerate-cpu.**
   `test_real_llama_bit_exact.py` is currently at 1B. At 70B it's a
   stronger claim and the gate for the paper team adopting fpwap.
4. **Peak VRAM as a function of N.** Their lmprobe-sweep OOM-killed at
   10k × 2048 during output materialization — the `[N, S, H×L]`
   landmine. fpwap's memmap-backed buffer should make peak VRAM flat
   in N. A plot demonstrates the spill contract works.
5. **Multi-dataset warm-start overhead.** 15 small sweeps back-to-back,
   one model_id, `Sweep → Sweep → …`. Report total wall-clock and the
   "setup per sweep" median. Zero = handle reuse works; non-zero =
   the integration ergonomics tax is quantified.

---

## In-flight

- [x] `scripts/benchmark.py --mode naive` (SPEC §17 ratio baseline; 8B shows 7.25× at 1024×128) — committed `d220b6b`
- [x] README 8B ratio row — committed `53dac5c` (but see note below; this is the framing the team cautioned against leading with)
- [ ] **`scripts/bench_n_scaling.py`** — sweeps N ∈ {8, 64, 512, 4096}
  at fixed seq_len, dumps CSV compatible with the team harness contract
  (model, mode, N, seq_len, microbatch, wall_s, tok_s, peak_gpu_mb,
  peak_cpu_mb, n_layers). Written, not yet committed, not yet run on
  hero. **This directly serves benchmark #1 and benchmark #4.**
- [ ] Run hero N-scaling on 70B + 8B; commit CSVs into `bench/results/`
  (directory tbd). Overnight-friendly at ~20 min per model.

## Queued — doable internally, high-leverage

- [ ] **Gap #4 — model-id → HF cache auto-resolve.** 15-min change.
  When `Sweep(model=str)` and the string isn't a path, call
  `snapshot_download(..., local_files_only=True)` internally; fall back
  to a clean error if the model isn't cached.
- [ ] **Gap #2 — `Extractor` handle for multi-sweep reuse.** Bigger
  API change; flagged, not unilaterally implemented. Shape TBD — need
  alignment. Draft API:
  ```python
  ext = fpwap.Extractor.from_hf("meta-llama/Llama-3.3-70B-Instruct")
  for dataset in datasets:
      sweep = ext.sweep(dataset=dataset, seq_len=256, callbacks=[...])
      result = sweep.run()
  # ext owns empty_model + accel_index + snapshot_dir; Sweep children reuse
  ```
- [ ] **Benchmark #4 dedicated** — fall out from `bench_n_scaling.py`
  if we also record `max_memory_allocated`; already wired above.

## Queued — needs coordination with the other repo

- [ ] **Gap #1 — `run_fpwap` in `benchmark_offload_backends.py`.**
  Draft locally as `scripts/harness_adapter.py` with the
  `(load_s, extract_times, fingerprint, peak_VRAM)` contract; hand to
  the team to drop into their harness.
- [ ] **Gap #5 — AUROC-parity test.** Lives in lmprobe. Out of scope
  for fpwap-solo work; coordinate with integration owner.
- [ ] **Benchmark #5 — multi-dataset warm-start overhead.** Tests gap #2
  after it lands.

## Deferred / scope-declined

- [ ] **Gap #3 — variable-length batches.** Genuinely architectural —
  requires rethinking the per-layer residual buffer shape ([N, S, H]
  assumes fixed S). Real workstream, not a quick fix. Park until the
  pad-to-max waste becomes the actual bottleneck.
- [ ] **Benchmark #3 — 70B bit-exactness.** Can't run on this box: 70B
  bf16 weights (~141 GB) exceed 128 GB RAM, so an `accelerate.cpu_offload`
  naive reference doesn't load. Possible on a larger machine.
- [ ] **Checkpoint/resume.** 18-min hero runs don't justify this yet.
- [ ] **NVMe-backed `ResidualBuffer`.** No concrete use case; current
  pinned-CPU buffer handles 10k × 128 @ 70B within 128 GB RAM.
- [ ] **Streaming-path `verify=True`.** Inherent limitation — the naive
  baseline needs the full model resident. Covered by preloaded verify.

## Reframe / cleanup

- [ ] **README Status table** currently leads with 8B-vs-naive (7.25×),
  which is the exact framing the team called "weak" — 8B is
  dense-fits-VRAM territory. Rework Status to lead with 70B-on-32GB,
  and fold the 8B ratio into a "baseline sanity" aside if kept at all.
- [ ] **`memory/spec17_ratio_demonstration.md`** uses the same 8B
  headline. Still valid as a regression gate, but not the headline
  claim. Note already acknowledges the hardware constraint on 70B.

---

## Notes for next operator

- `scripts/benchmark.py --mode naive` requires the model to fit in CPU
  RAM for `accelerate.cpu_offload`. 70B on 128 GB RAM will OOM; use
  `accelerate-disk-stream` equivalent (not wired here) for 70B naive.
- `torch.cuda.reset_peak_memory_stats()` is called per-N in
  `bench_n_scaling.py` so each row's `peak_gpu_mb` is independent.
  `ru_maxrss` is process-lifetime so `peak_cpu_mb` is monotonic —
  interpret as "RAM high-water so far."
- The existing integration recipe memory (`integration_recipe.md`) —
  `from_fpwap()` classmethod at the activation-source dispatch layer —
  is still the shape for gap #2 consumption; that memory captured the
  past lmprobe leak-through experience.
