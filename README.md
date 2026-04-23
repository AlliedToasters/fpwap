# fpwap — Forward Pass Weight Amortization Protocol

A single-purpose library for running activation extraction over large transformer models **whose weights don't fit in your GPU**, across datasets of **thousands of prompts**, on **consumer hardware**, at **full precision**.

## The regime

You're a mech-interp researcher. Your model is bigger than your VRAM. Your dataset is thousands of prompts. Adjacent tools each fail in a way that changes what you're studying:

- **Quantization** (bitsandbytes, GPTQ) changes the activations you're trying to read.
- **Inference servers** (vLLM, TGI) optimize next-token throughput, not residual-stream extraction.
- **`accelerate.cpu_offload`** streams weights once per batch — 10k prompts × 80 layers on a 70B model is hundreds of TB of weight I/O, hours of wall-clock per dataset pass.
- **Cloud GPUs** break your interactive iteration loop and cost hundreds per experiment.

fpwap inverts the inference loop: **load each layer once, stream the whole dataset through it**, spill intermediates to disk, move on. Total weight I/O drops from `O(N_batches × N_layers)` to `O(N_layers)`. A 10k-sample Llama-3.1-70B extraction on a 32 GB consumer GPU runs in roughly the wall-clock of a single batch under the naive approach — with the same weights, no quantization, no cloud.

## Aspirational performance

Targets, not measurements. These are the numbers fpwap is being built to; each row unlocks only after its milestone lands (70B gates on the bit-perfect test; 405B gates on the mmap-from-HF-cache path). Replaced by measured benchmarks as they come in.

### Reference machine

| Component | Spec |
| --------- | ---- |
| GPU       | NVIDIA RTX 5090, 32 GB VRAM |
| CPU       | Modern desktop-class, 16+ cores |
| RAM       | 128 GB DDR5 |
| Storage   | NVMe SSD (Gen 4+), ≥ 1 TB free |
| Interconnect | PCIe 5.0 x16 |
| Network   | None — fully local, no cloud |

### Dataset-scale activation extraction (10,000 prompts × 256 tokens = 2.56M tokens)

Residual stream (`residual_post`) captured at every layer, pooled to last token, persisted to disk. `RawActivations(layers="all")`.

| Model | Weights (bf16) | Loading strategy | Wall-clock target | Throughput target | vs. naive `accelerate.cpu_offload` |
| ----- | -------------- | ---------------- | ----------------- | ----------------- | ----------------------------------- |
| Llama-3.1-8B   | 16 GB   | `cpu_offload`      | ≤ 8 min  | ≥ 5,000 tok/s | ≥ 4× faster |
| Llama-3.1-70B  | 140 GB  | `disk_offload`     | ≤ 45 min | ≥ 950 tok/s   | ≥ 4× faster (naive ≈ 3 h) |
| Llama-3.1-405B | 810 GB  | `mmap_from_cache`  | ≤ 4 h    | ≥ 180 tok/s   | naive infeasible (OOM in RAM) |

Throughput is end-to-end tokens per second — total tokens processed (samples × seq_len) divided by wall-clock from `fpwap(...).run()` entry to return, including weight I/O, forward, callbacks, and buffer write.

### Single-pass cost per layer (Llama-3.1-70B, 1.75 GB weights per layer)

The inner loop that fpwap is optimizing. On the reference machine, per layer, per full sweep of 10k × 256-token samples:

| Phase | Budget | Notes |
| ----- | ------ | ----- |
| Weight load  | ≤ 1.0 s    | NVMe → CPU → GPU, `disk_offload` path; once per layer, not once per batch |
| Forward      | ≤ 15 s     | 10k samples, bf16, batched at engine's discretion |
| Callback     | ≤ 1.0 s    | Aggregate across all registered callbacks for this layer |
| Buffer write | ≤ 1.0 s    | Pooled activations to memmap; raw `[N, S, H]` budget is higher |
| **Per-layer total** | **≤ 18 s** | × 80 layers ≈ 24 min (leaves headroom vs. 45 min end-to-end target) |

### Overhead budgets

| Surface | Budget | Why |
| ------- | ------ | --- |
| Profile + progress, combined | < 1% wall-clock | Has to stay on by default — see the [Observability](#observability) section |
| `verify=True` (vs. naive `cpu_offload` at every layer) | 2–3× slower | Correctness debugging only; not for production runs |
| Preflight | < 5 s | Rejects infeasible configurations before GPU contact |

## The API

One verb. One callback class. One result.

```python
from fpwap import fpwap
from fpwap.callbacks.common import RawActivations, IncrementalPCA, DiffOfMeans

run = fpwap(
    model="meta-llama/Llama-3.1-70B",
    dataset=my_dataset,                # iterable of {"input_ids": ..., "label": ...}
    seq_len=256,
    callbacks=[
        RawActivations(layers=[40, 45, 50]),               # pooled by default
        IncrementalPCA(layers="all", n_components=64),
        DiffOfMeans(layers="all", label_fn=lambda s: s["label"]),
    ],
)

plan = run.preflight()
print(plan.summary())                   # check feasibility before GPU contact

result = run.run()
acts  = result.activations(layer=45, hook="residual_post")   # [N, H]
basis = result.artifact("pca_basis", layer=45)
```

That is the entire user-facing surface for read-only workflows. No backend objects to construct. No `batch_size` knob to foot-gun. No `loader` / `accumulator` triple to wire up. Construction is cheap; `.preflight()` inspects the plan and rejects infeasible configurations with actionable messages; `.run()` executes.

### Layer indexing

Hook names follow the HF `hidden_states` convention:

| Hook | Equals |
| ---- | ------ |
| `residual_pre` at layer `L`  | `hidden_states[L]`   (input to block `L`) |
| `residual_post` at layer `L` | `hidden_states[L+1]` (output of block `L`) |
| `attn_out` at layer `L`      | attention sub-layer output at block `L` |
| `mlp_out` at layer `L`       | MLP sub-layer output at block `L` |

No off-by-one translation at the call site.

### Writing your own callback

Subclass `fpwapCallback`. Declare which layers and hooks you want; implement `on_batch`. Return an `Emit` to persist a tensor, a `WriteBack` to modify the residual before the next layer, or `None` to no-op.

```python
from fpwap import fpwapCallback, Emit

class LastTokenLogNorm(fpwapCallback):
    target_layers = [32]
    target_hooks = ("residual_post",)
    phase = "read"

    def on_batch(self, layer_idx, hook, acts, sample_ids):
        return Emit(acts[:, -1, :].norm(dim=-1).log())
```

### Write-backs and multi-pass workflows

The same entry point handles steering. A callback with `phase = "write"` modifies the residual stream between layers; artifacts from one run feed the next.

```python
from fpwap.callbacks.common import SteerInBasis

# Pass 2: steer in the basis fit during pass 1
steer = fpwap(
    model="meta-llama/Llama-3.1-70B",
    dataset=my_dataset,
    seq_len=256,
    callbacks=[
        SteerInBasis(
            basis_artifact=result.artifact("pca_basis", layer=45),
            direction_idx=0,
            alpha=2.0,
            layers=[45],
        ),
    ],
)
steered = steer.run()
```

## Observability

Performance is the product, so every run is profiled by default with a measurement overhead small enough (target: under 1% wall-clock) that you never have to opt in. When a run is slower than you want, the answer is already in `result.profile` — no re-running with `profile=True`.

```python
result = run.run()

result.profile.summary()            # human-readable breakdown per layer
result.profile.by_phase()           # load / forward / callback / write
result.profile.slowest_layer()      # where the time went
result.profile.bytes_moved()        # weight I/O, buffer I/O
```

Interactive progress is on by default — a tqdm-style bar across layers × batches, because a run on the workstation under your desk should not sit silent for 40 minutes. Disable with `progress=False`; pass a callable (`progress=my_reporter`) to stream events into wandb, rich, or any other backend.

## Reference callbacks

Four callbacks ship with the library as examples and integration tests:

- **`RawActivations`** — persist per-sample activations, pooled (`last_token_only=True`) by default to avoid an `[N, S, H]` memory landmine.
- **`IncrementalPCA`** — fit a PCA basis over the entire dataset in a single pass.
- **`DiffOfMeans`** — compute per-class activation means for binary-labeled data.
- **`SteerInBasis`** — additive intervention in a pre-computed basis; `phase = "write"`.

Anything beyond these four is a consumer's problem.

## Integrating fpwap into a research codebase

The recommended shape is a single classmethod on your codebase's activation-source type, inserted **above** any per-batch sharding your framework does:

```python
class Activations:
    @classmethod
    def from_fpwap(cls, model_id, prompts, layers, pool="last_token"):
        run = fpwap(
            model=model_id,
            dataset=_as_dataset(prompts),
            seq_len=...,
            callbacks=[
                RawActivations(
                    layers=layers,
                    last_token_only=(pool == "last_token"),
                ),
            ],
        )
        return cls.from_result(run.run())
```

Branch `use_fpwap` at your dispatch layer — the same place you'd branch between `from_model`, `from_goodfire`, etc. — not inside a per-batch loop. fpwap's value (amortizing layer loads across the whole dataset) only materializes if it sees the dataset; if your framework shards externally and calls an extractor per shard, lift the dispatch up one level before integrating.

## Scope

fpwap is a plumbing layer. It produces activations and accepts transforms. It does not know what a probe is. Linear probe fitting, SAE training, attribution analysis, and any other statistical modeling of activations belong in consumer libraries. If it requires knowing what a probe is, it's out of scope.

## Status

Early skeleton — the engine is not yet implemented. The bit-perfect test at `tests/gpu/test_forward_bit_perfect.py` (fpwap vs. `accelerate.cpu_offload` residual match at every layer) is the definition of "done" for the first milestone.

See `SPEC.md` for the full design.
