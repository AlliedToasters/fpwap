# `fpwap` — Forward Pass Weight Amortization Protocol

**Status:** Draft · design spec, not yet implemented
**Working name:** `fpwap`

---

## 1. Motivation

Accelerate's `cpu_offload` and `disk_offload` solve per-forward weight streaming well. Call `model(inputs)` normally; `AlignDevicesHook`s installed on every submodule stream weights onto the GPU as each submodule enters forward, and evict them on exit. The GPU never holds more than a few layers at a time, transfers are pinned and overlapped with compute, and the user writes normal PyTorch.

What accelerate does not do is amortize the streaming cost across a dataset. A 2,500-batch pass through Llama-70B still calls 2,500 forwards, each of which restreams ~140 GB of weights — ~350 TB total. On a retail rig with a PCIe 4.0 link, wall-clock is bandwidth-bound for the entire run.

**fpwap inverts the loop.** Instead of "for each batch, do a full forward," it is "for each layer, stream the whole dataset through it." Load layer `N` once, run every microbatch through it, write the post-layer residual to NVMe, advance to layer `N+1`. Weight I/O drops from `O(N_batches × N_layers)` to `O(N_layers)`. For a 10k-sample run through 70B, that is ~140 GB of weight streaming total — roughly the cost of one naive batch.

**The trade:** one residual-stream buffer persisted between layers — `N_samples × seq × hidden × dtype_bytes`. At seq=256, hidden=8192, bf16, 10k samples: ~40 GB. Cheap on a big-NVMe workstation, impossible on a GPU-only cloud box. This library assumes the former.

**Mental model.** Accelerate is the weight-movement layer. fpwap is the dataset-amortization layer on top. fpwap depends on accelerate's primitives (`OffloadedWeightsLoader`, `AlignDevicesHook`, `dispatch_model`) for the actual transport; it does not reimplement them. fpwap's contribution is the inverted loop, the residual buffer, the callback protocol, and the checkpoint/resume machinery.

## 2. Scope

### In scope

- Single-node, single-GPU execution on dense transformer architectures.
- Fixed-length forward passes: encoder-style inference where `seq_len` is known per batch and attention is non-causal-dependent across samples.
- Residual-stream-as-transport between layers.
- Callback-driven extraction, transformation, and residual write-back.
- Pluggable storage backends (default: NVMe memmap + parquet index).
- Checkpoint / resume at layer boundaries.
- Multi-pass workflows where pass-N artifacts feed pass-(N+1) callbacks.
- **Models larger than CPU RAM** via the mmap-from-HF-cache path (§12.3).

### Out of scope

- Autoregressive generation. KV-cache dependencies across time steps break loop inversion. Explicitly unsupported; preflight should detect and reject.
- Training-time gradient flow. Deferred (see §13). API should not preclude.
- Multi-GPU tensor/pipeline parallelism. Deferred.
- Non-transformer architectures.
- MoE with per-layer expert sharding across devices. (Single-device MoE is fine — experts are just part of the layer's weight set.)

### Non-goals

- Replacing accelerate or HuggingFace. fpwap is built *on* accelerate. Any weight-movement feature that already exists in accelerate should be used directly, not wrapped.
- Being a probing library. Probing is a downstream consumer concern. fpwap produces activations and accepts transforms; it does not know what a probe is.

### Dependencies

- `accelerate` >= 1.0 — `OffloadedWeightsLoader`, `AlignDevicesHook`, `dispatch_model`, `init_empty_weights`, `cpu_offload`, `disk_offload`.
- `safetensors` — mmap-backed weight access via `safe_open`.
- `torch` >= 2.1 — bf16 transport, memmap integration.
- `transformers` — for model loading (users supply the model class).
- `pyarrow` — parquet index for the default storage backend.

## 3. Core Abstractions

### 3.1 `fpwap`

The engine. Takes a model (or model spec), a dataset, a list of callbacks, a storage backend, and a dtype policy. Executes the inverted loop. Single entry point: `fpwap.run() -> fpwapResult`.

### 3.2 `fpwapCallback`

Base class users subclass. Declares which layers/hooks it targets, which phase it runs in, and implements one or more lifecycle methods. See §5.

### 3.3 `StorageBackend`

Interface for writing per-sample outputs. Default implementation is memmap + parquet index. Alternate backends: sharded safetensors, zstd-compressed numpy.

### 3.4 `ResidualBuffer`

The inter-layer transport. Backed by a single NVMe memmap sized `N_samples × seq × hidden × dtype_transport`. Owned by the `fpwap` engine. Write-back callbacks mutate it in place; read callbacks see the current state.

### 3.5 `fpwapArtifact`

The output of a run-scoped or layer-scoped callback. Named, addressable by `(fpwap_id, layer_idx, hook, kind)`. Can be passed as input to a subsequent run. Examples: fitted PCA basis per layer, difference-of-means vector per layer, steering vector.

### 3.6 `Preflight`

A planner that, given `(model, dataset, seq_len, vram_budget, nvme_free, callbacks)`, returns a feasibility report and a concrete execution plan (microbatch size, residual buffer size, per-layer peak VRAM, estimated wall-clock). Runs in seconds.

### 3.7 Weight movement — no fpwap-specific abstraction

**fpwap does not define a `LayerLoader` or similar.** Weight movement is `accelerate.utils.OffloadedWeightsLoader` plus direct parameter assignment, wrapped in two small helpers (`_load_layer`, `_unload_layer`) that live in `fpwap.loader`. These are implementation details, not public API. See §12.

## 4. The Loop

```python
preflight(model_spec, dataset, callbacks) -> plan
loader = build_offloaded_weights_loader(model_spec)   # §12
model  = init_empty_model(model_spec)                 # weights on "meta"
residual_buffer = allocate_memmap(N_samples, seq, hidden, dtype_transport)
residual_buffer[:] = run_embedding(model, dataset, loader)  # pass 0

for layer_idx in range(N_layers):
    _load_layer(model, layer_idx, loader)             # direct param assignment
    for cb in callbacks: cb.on_layer_start(layer_idx)

    for microbatch, sample_ids in microbatches(residual_buffer, plan.microbatch_size):
        acts = model.layers[layer_idx](microbatch)    # no AlignDevicesHook firing
        for (hook, phase) in active_hooks_at(layer_idx):
            run_read_callbacks(acts, layer_idx, hook, sample_ids)
            acts = run_write_callbacks(acts, layer_idx, hook, sample_ids)
            run_read_after_write_callbacks(acts, layer_idx, hook, sample_ids)
        residual_buffer[sample_ids] = acts

    for cb in callbacks: cb.on_layer_end(layer_idx)
    _unload_layer(model, layer_idx)
    checkpoint(layer_idx, residual_buffer, callback_states)

for cb in callbacks: cb.on_fpwap_end()
return fpwapResult(artifacts, storage_handles)
```

**Key subtlety.** The inner loop calls `model.layers[layer_idx](microbatch)` directly, with no `AlignDevicesHook` firing on forward. Hooks are accelerate's mechanism for per-forward streaming — that is the behavior fpwap exists to avoid. fpwap manages load/unload cadence manually (once per layer, not once per forward) via `_load_layer` and `_unload_layer`.

## 5. Callback Interface

### 5.1 Declaration

```python
class fpwapCallback:
    target_layers: Sequence[int] | Literal["all"] = "all"
    target_hooks: Sequence[HookName] = ("residual_post",)
    phase: Literal["read", "write", "read_after_write"] = "read"
    needs_grad: bool = False  # reserved for future; must be False

    def on_fpwap_start(self, ctx: fpwapContext) -> None: ...
    def on_layer_start(self, layer_idx: int) -> None: ...
    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,                    # (microbatch, seq, hidden), on GPU
        sample_ids: Tensor,              # (microbatch,), int64
    ) -> BatchResult | None: ...
    def on_layer_end(self, layer_idx: int) -> LayerArtifact | None: ...
    def on_fpwap_end(self) -> fpwapArtifact | None: ...
    def checkpoint_state(self) -> bytes: ...
    def restore_state(self, state: bytes) -> None: ...
```

`BatchResult` is one of:
- `Emit(tensor, dtype=None)` — persist per-sample outputs via the storage backend.
- `WriteBack(tensor)` — replace the residual slice. Only legal when `phase == "write"`.
- `None` — callback ran but produced nothing this batch (stateful accumulation).

### 5.2 Phase ordering

Within a single `(layer_idx, hook)` trigger:

1. All `read` callbacks, in registration order. See pre-write residual. Must not return `WriteBack`.
2. All `write` callbacks, composed (see §5.3). Produce post-write residual.
3. All `read_after_write` callbacks, in registration order. See post-write residual.

Registration order within a phase should not be load-bearing. If a callback's correctness depends on another callback running first, encode that as an explicit dependency, not an ordering hack.

### 5.3 Write composition

When multiple `write` callbacks target the same `(layer, hook)`:

- **Default:** sequential. Each callback's `WriteBack` becomes the input to the next. Warning logged on first use.
- **Opt-in additive:** register callbacks in a `WriteGroup(mode="additive")`. Each callback receives the original pre-write residual; deltas are summed.
- **Disjoint sample writes:** no conflict.

Additive is right for "sum of steering vectors." Sequential is right for "project out direction D, then add vector V." Users will want both; make the mode explicit.

### 5.4 The four canonical patterns

1. **Stateless per-batch emit.** Raw activation extraction, frozen probe application, projection onto a precomputed basis. `on_batch` → `Emit`.
2. **Per-batch emit + per-layer finalize.** Streaming DoM, per-layer moments. `on_batch` → `Emit`; `on_layer_end` → `LayerArtifact`.
3. **Per-batch accumulate, end emit.** Incremental PCA, quantile sketches, activation atlases. Stateful `on_batch`; `on_fpwap_end` → `fpwapArtifact`. No per-sample storage.
4. **Write-back (steering, ablation).** `on_batch` → `WriteBack`; `phase = "write"`.

The engine materializes only the hooks that have at least one callback registered. A layer with no callbacks is a pure transport step.

## 6. Targeting and Hook Taxonomy

### 6.1 Hook names

- `residual_pre` — residual entering the layer (post-embedding for layer 0).
- `residual_post` — residual leaving the layer (what goes into `residual_buffer`).
- `attn_out` — attention sublayer output, pre-residual-add.
- `mlp_out` — MLP sublayer output, pre-residual-add.

Only `residual_post` is guaranteed across all supported architectures. Others require model-specific hook plumbing and may be opt-in per model family.

### 6.2 Target resolution

The engine takes the union of `(layer_idx, hook)` across all registered callbacks and materializes exactly that set. A hook with no subscribers is never computed.

## 7. Storage Backend

### 7.1 Interface

```python
class StorageBackend(Protocol):
    def open_shard(self, fpwap_id: str, layer_idx: int, hook: HookName,
                   kind: str, schema: ActivationSchema) -> ShardHandle: ...
    def write_rows(self, handle: ShardHandle, rows: Tensor,
                   sample_ids: Tensor) -> None: ...
    def close_shard(self, handle: ShardHandle) -> ShardManifest: ...
    def write_artifact(self, fpwap_id: str, key: ArtifactKey,
                       artifact: fpwapArtifact) -> None: ...
```

### 7.2 Default backend

Memmap + parquet index:

- One parquet manifest per `(fpwap, layer, hook, kind)` with `(sample_id, shard_index, row_offset)`.
- Sharded safetensors or raw memmap for activations.
- Lazy progressive read on the consumer side.

### 7.3 Alternate backends

- **Safetensors-sharded** — for publishing to HF Hub.
- **Zstd-numpy** — for compressible content (logits, one-hots).
- **Null** — for write-back-only runs; no storage cost.

## 8. Dtype Policy

### 8.1 Transport

`residual_buffer` is bf16 by default. Override to fp16 or fp32 per run. Match the model's native dtype to avoid per-batch casting.

### 8.2 Statistical accumulation

Any callback that accumulates statistics across samples **must** promote to fp32 internally. Enforced by convention. The `fpwapCallback` base class exposes `self.accum_dtype = torch.float32` as a default; callbacks should use it. Silent bf16 drift over 10k samples is a real bug that reads as a modeling issue.

### 8.3 Per-callback override

Emit callbacks can specify output dtype independently of transport dtype. A DoM callback can write fp16 means without affecting the residual buffer.

## 9. Checkpoint and Resume

### 9.1 Checkpoint granularity

Layer boundary. After `on_layer_end`:

- `residual_buffer` is a memmap; fsync.
- `fpwap_state.json`: `last_completed_layer`, callback state hashes, preflight plan, dataset fingerprint.
- Per-callback serialized state via `fpwapCallback.checkpoint_state() -> bytes`.

### 9.2 Resume protocol

On `fpwap.run()` startup, if `fpwap_state.json` exists and matches `(model_id, dataset_fingerprint, callback_signature)`:

1. Load `residual_buffer` memmap.
2. Restore callback state via `fpwapCallback.restore_state(bytes)`.
3. Jump to `last_completed_layer + 1`.

Any mismatch → refuse to resume with a clear diff.

### 9.3 Atomicity

A layer is either fully committed or not. Partial commits rolled back. Write to `.tmp`, atomic rename on success.

## 10. Preflight Planner

### 10.1 Inputs

- `model_spec` (config + weight location, typically an HF snapshot dir)
- `dataset_size`, `seq_len`
- `vram_budget_gb`, `nvme_free_gb`, `cpu_ram_gb`
- `callbacks`
- `transport_dtype`

### 10.2 Outputs

```python
@dataclass
class PreflightReport:
    feasible: bool
    blockers: list[str]
    microbatch_size: int
    residual_buffer_gb: float
    per_layer_peak_vram_gb: float
    estimated_wall_clock_s: float
    estimated_weight_io_gb: float
    warnings: list[str]
    loading_strategy: Literal["cpu_offload", "disk_offload", "mmap_from_cache"]
```

### 10.3 Estimation approach

Microbatch size by binary search on a dry-run single layer. Wall-clock by `N_layers × (load_time + microbatches × per_microbatch_fwd)`. Per-layer peak VRAM from static analysis of parameter count + activation footprint. Loading strategy selection per §12.2.

This is the marketing demo. "Here's your 70B run, it will finish in 47 minutes, it will produce 120 GB of activations, you have 800 GB free, press enter."

## 11. Multi-pass Workflows

### 11.1 Artifacts as inputs

```python
pca_fpwap = fpwap(model, dataset, callbacks=[IncrementalPCAFit(n_components=64)])
pca_result = pca_fpwap.run()

steer_fpwap = fpwap(
    model, dataset,
    callbacks=[SteerInBasis(
        basis=pca_result.artifact("pca_basis"),
        direction_idx=3,
        alpha=2.0,
    )],
)
```

### 11.2 Artifact addressing

`(fpwap_id, layer_idx, hook, kind)`. `fpwap_id` is explicit or content-addressed hash of `(model_id, dataset_fingerprint, callback_signature)`.

### 11.3 Composition

Pass 1 fits per-layer PCA. Pass 2 computes probe readouts post-ablation of a single PC. Pass 3 measures downstream effect on the residual stream N layers later. Each pass independent; storage backend is the integration point.

## 12. Model Loading via Accelerate

fpwap uses three accelerate loading strategies depending on where the model fits, selected automatically by preflight.

### 12.1 Three regimes

| Strategy | Right when |
|---|---|
| `cpu_offload(model, execution_device=cuda:0)` | Model fits in CPU RAM. Fastest per-forward streaming. For fpwap: used with manual load/unload cadence, not per-forward hooks. |
| `disk_offload(model, offload_dir)` | Model fits in CPU RAM, run is long-running, NVMe offload amortizes reads. |
| mmap-from-HF-cache (§12.3) | Model *does not* fit in CPU RAM. Strictly more general than the above two; works for any model that fits on disk. |

### 12.2 Strategy selection (preflight)

```
if model_size_bytes <= cpu_ram_budget * 0.7:
    strategy = "cpu_offload"
elif model_size_bytes <= cpu_ram_budget * 1.5 and nvme_free > 2 * model_size_bytes:
    strategy = "disk_offload"
else:
    strategy = "mmap_from_cache"
```

Thresholds are conservative; users can override.

### 12.3 The mmap-from-HF-cache path

For models larger than CPU RAM, neither `cpu_offload` nor `disk_offload` works out of the box:

- `disk_offload(model, offload_dir)` requires `model.state_dict()` materialized in RAM for its initial copy. Host OOM on an 810 GB model / 128 GB box.
- `AutoModelForCausalLM.from_pretrained(..., device_map="auto", max_memory={tight})` raises `ValueError: use disk_offload instead`.

But `disk_offload` has a quiet escape hatch: **if `offload_dir/index.json` already exists, it skips the materializing copy and attaches hooks directly.** And `OffloadedWeightsLoader` already supports index entries pointing at HF-cache safetensors files (mmap-backed via `safe_open`). The recipe:

```python
from accelerate import init_empty_weights, disk_offload

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
model.tie_weights()  # CRITICAL — must be before index construction

accel_index = build_accel_index_from_hf_cache(snapshot_dir)
alias_tied_weights_in_index(model, accel_index)

with open(offload_dir / "index.json", "w") as f:
    json.dump(accel_index, f)                       # MUST be before disk_offload

model = disk_offload(
    model,
    offload_folder=str(offload_dir),
    execution_device=torch.device("cuda:0"),
)
```

Full implementation in Appendix C. This should be exposed as `fpwap.loader.load_from_cache(model_id, offload_dir)` — a public helper, because nothing in accelerate exposes it today (see huggingface/accelerate#4016).

**Result:** the model never lands in CPU RAM. A 405B forward pass on 128 GB RAM / 32 GB VRAM peaks at ~4 GB VRAM with steady-state disk reads.

### 12.4 Bypassing `AlignDevicesHook` for fpwap

Accelerate's loading strategies all install `AlignDevicesHook`s that stream weights per-forward. That is the naive behavior fpwap exists to avoid. The fpwap engine uses one of two approaches:

**Approach A (preferred):** use `OffloadedWeightsLoader` directly, assign parameters manually, never install `AlignDevicesHook`.

```python
def _load_layer(model, layer_idx, loader):
    layer = model.model.layers[layer_idx]
    for param_name, param in layer.named_parameters():
        full_name = f"model.layers.{layer_idx}.{param_name}"
        param.data = loader[full_name].to("cuda:0", non_blocking=True)

def _unload_layer(model, layer_idx):
    layer = model.model.layers[layer_idx]
    for param in layer.parameters():
        param.data = torch.empty(0, device="meta")
```

**Approach B (fallback):** install hooks but manually trigger pre-hook once per layer, suppress post-hook until the fpwap window closes. More fragile; use only if Approach A has an unforeseen blocker.

fpwap implements Approach A.

### 12.5 What fpwap does NOT reimplement from accelerate

- Pinned host memory and async H2D transfer — accelerate's `OffloadedWeightsLoader` handles this.
- `safe_open` handle caching — `OffloadedWeightsLoader` keeps these alive internally.
- Stream synchronization for compute/transfer overlap — built into accelerate's hook path (referenced for non-fpwap forwards; fpwap's manual path uses `non_blocking=True` and a single CUDA stream initially, adding multi-stream overlap if benchmarks justify).
- Per-forward streaming for non-fpwap calls — if users want to call `model(x)` outside of fpwap (e.g. for a quick eval), accelerate's standard offload paths work as normal.

## 13. Deferred: Training Mode

Not yet implemented, but the API should not preclude:

- `fpwapCallback.needs_grad: bool`.
- Engine tracks whether any callback at layer `L` needs grads. If yes, retain graph for that layer; otherwise `no_grad`.
- A "trainable head" pattern: small GPU-resident module, receives activations, accumulates loss, backprops at `on_layer_end`.
- Write-back with grad is *not* supported even in principle — turns fpwap into a backward pass, separate design exercise.

Design question: does the current API require the engine to know about optimizers, or can the callback own its optimizer entirely? Preference: callback owns optimizer, engine owns scheduler of grad-enabled layers.

## 14. Error Handling and Debugging

### 14.1 Deterministic ordering

Same `(model, dataset, seed, callback_signature)` → bit-identical outputs. Easier for fpwap than for naive training loops: no per-epoch shuffling, dataset walked once per layer in fixed order.

### 14.2 Verification mode

`fpwap(..., verify=True)` runs the first `K` samples through both the naive accelerate `cpu_offload` path and the inverted fpwap path and diffs `residual_post` at every layer. Tolerance configurable per dtype. Slow, but invaluable during development and when adding hooks.

### 14.3 Observability

Per-layer timings, I/O bytes, callback time. Structured log output plus a final report. When a user asks "why is my fpwap slow," the answer is in the report.

## 15. Public API Sketch

```python
from fpwap import fpwap, fpwapCallback, Emit, WriteBack
from fpwap.storage import MemmapBackend
from fpwap.loader import load_from_cache
from fpwap.callbacks.common import RawActivations, IncrementalPCA, DiffOfMeans

run = fpwap(
    model="meta-llama/Llama-3.1-70B",
    dataset=my_dataset,
    seq_len=256,
    callbacks=[
        RawActivations(layers=[40, 45, 50], hook="residual_post"),
        IncrementalPCA(layers="all", n_components=64),
        DiffOfMeans(layers="all", label_fn=lambda s: s["label"]),
    ],
    storage=MemmapBackend("/data/fpwaps/run_01"),
    transport_dtype=torch.bfloat16,
    # loading_strategy inferred by preflight; override with:
    # loading_strategy="mmap_from_cache",
)

plan = run.preflight()
print(plan.summary())

result = run.run()
result.artifact("pca_basis", layer=45)
result.activations(layer=45, hook="residual_post")
```

## 16. Open Questions

1. **Attention patterns.** Different shape, different storage. Deferred if demanded.
2. **Dataset contract.** Assumes yields `(input_ids, attention_mask, sample_id, metadata)`. Metadata opaque to engine, available to callbacks via `MetadataAccessor` injected at `on_fpwap_start`.
3. **Partial write-back.** Can a write-back callback modify a subset of samples? Current design: full tensor, unmodified samples return `acts_in` unchanged.
4. **Determinism across hardware.** bf16 matmul is not bitwise identical across GPUs. Verification mode tolerance must be aware. Document per (architecture, dtype).
5. **Multi-stream transfer overlap.** Approach A (§12.4) uses a single CUDA stream. If benchmarks show bandwidth is not saturated, add a second stream for weight prefetch of layer `N+1` during layer `N` compute.

## 17. Success Criteria

Ships when:

- A 10k-sample run through Llama-3.1-70B on a 32 GB GPU completes in **≤ 25% of the wall-clock** of the `accelerate.cpu_offload` naive baseline, and produces activations identical within bf16 tolerance.
- A 10k-sample run through Llama-3.1-405B on 128 GB RAM / 32 GB VRAM completes at all via the mmap-from-HF-cache path.
- Mid-run kill resumes correctly.
- Four reference callbacks (raw extraction, incremental PCA, DoM, steering) each <200 LoC.
- Preflight rejects infeasible configurations with clear, actionable messages before touching GPU.

---

## Appendix A — Callback implementation examples

### Raw activations

```python
class RawActivations(fpwapCallback):
    phase = "read"

    def __init__(self, layers, hook="residual_post", last_token_only=True):
        self.target_layers = layers
        self.target_hooks = (hook,)
        self.last_token_only = last_token_only

    def on_batch(self, layer_idx, hook, acts, sample_ids):
        if self.last_token_only:
            acts = acts[:, -1, :]
        return Emit(acts, dtype=torch.bfloat16)
```

### Streaming difference of means

```python
class DiffOfMeans(fpwapCallback):
    phase = "read"
    target_hooks = ("residual_post",)

    def __init__(self, layers, label_fn):
        self.target_layers = layers
        self.label_fn = label_fn
        self.sum_pos = defaultdict(lambda: 0)
        self.sum_neg = defaultdict(lambda: 0)
        self.n_pos = defaultdict(int)
        self.n_neg = defaultdict(int)

    def on_batch(self, layer_idx, hook, acts, sample_ids):
        labels = self.label_fn(sample_ids)
        last = acts[:, -1, :].to(torch.float32)  # accum in fp32
        pos_mask = labels == 1
        self.sum_pos[layer_idx] += last[pos_mask].sum(0)
        self.n_pos[layer_idx] += pos_mask.sum().item()
        self.sum_neg[layer_idx] += last[~pos_mask].sum(0)
        self.n_neg[layer_idx] += (~pos_mask).sum().item()
        return None

    def on_layer_end(self, layer_idx):
        mu_pos = self.sum_pos[layer_idx] / max(self.n_pos[layer_idx], 1)
        mu_neg = self.sum_neg[layer_idx] / max(self.n_neg[layer_idx], 1)
        return LayerArtifact("dom_vector", (mu_pos - mu_neg).to(torch.bfloat16))
```

### Steering in PC basis (two-pass)

```python
class SteerInBasis(fpwapCallback):
    phase = "write"
    target_hooks = ("residual_post",)

    def __init__(self, basis_artifact, direction_idx, alpha, layers):
        self.target_layers = layers
        self.basis = basis_artifact
        self.direction_idx = direction_idx
        self.alpha = alpha

    def on_batch(self, layer_idx, hook, acts, sample_ids):
        v = self.basis[layer_idx][:, self.direction_idx]
        delta = self.alpha * v
        return WriteBack(acts + delta)
```

## Appendix B — What this spec deliberately does not contain

- A probe abstraction. Probing is a downstream consumer concern; fpwap produces activations and accepts transforms.
- A dataset loader. Datasets are user-supplied and conform to a minimal protocol.
- A serving or API layer.
- Opinions on activation visualization.

The smallest thing that solves the amortization problem with enough abstraction to be useful for ~five concrete workloads. Anything else is scope creep and belongs in a consumer library.

## Appendix C — The mmap-from-HF-cache recipe

Reference implementation for `fpwap.loader.load_from_cache`. This is the path that makes 405B-on-128GB-RAM possible. Not exposed by accelerate as a public helper today (see huggingface/accelerate#4016).

```python
from pathlib import Path
import json
from accelerate import init_empty_weights, disk_offload
from accelerate.utils import OffloadedWeightsLoader
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
import torch

_SAFE_TO_TORCH_DTYPE = {
    "F64": "float64", "F32": "float32", "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}


def build_accel_index_from_hf_cache(snapshot_dir: Path) -> dict:
    """Convert HF's model.safetensors.index.json → accelerate's loader format."""
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        hf_index = json.load(f)
    accel_index = {}
    for weight_name, shard_file in hf_index["weight_map"].items():
        shard_path = str(snapshot_dir / shard_file)
        with safe_open(shard_path, framework="pt") as st:
            sl = st.get_slice(weight_name)
            accel_index[weight_name] = {
                "safetensors_file": shard_path,
                "weight_name": weight_name,
                "dtype": _SAFE_TO_TORCH_DTYPE[sl.get_dtype()],  # critical — see Appendix D gotcha 1
                "shape": list(sl.get_shape()),
            }
    return accel_index


def alias_tied_weights_in_index(model, accel_index: dict) -> None:
    """Detect tied weights by object identity and add aliases to the index.
    
    Requires model.tie_weights() called beforehand.
    Requires remove_duplicate=False to see aliases.
    """
    by_id: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        by_id.setdefault(id(param), []).append(name)
    for names in by_id.values():
        sources = [n for n in names if n in accel_index]
        targets = [n for n in names if n not in accel_index]
        if sources and targets:
            for t in targets:
                accel_index[t] = accel_index[sources[0]]


def load_from_cache(
    model_id: str,
    snapshot_dir: Path,
    offload_dir: Path,
    execution_device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
):
    """Load a model larger than CPU RAM via mmap-from-HF-cache.
    
    Model is never materialized in RAM. Weights are mmap'd from the
    safetensors files in the HF cache.
    """
    config = AutoConfig.from_pretrained(model_id)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.tie_weights()  # MUST be before index construction

    accel_index = build_accel_index_from_hf_cache(snapshot_dir)
    alias_tied_weights_in_index(model, accel_index)

    offload_dir.mkdir(parents=True, exist_ok=True)
    with open(offload_dir / "index.json", "w") as f:
        json.dump(accel_index, f)  # MUST be before disk_offload

    model = disk_offload(
        model,
        offload_folder=str(offload_dir),
        execution_device=execution_device,
    )
    return model
```

fpwap uses the underlying `OffloadedWeightsLoader` (constructed from `accel_index`) directly for its manual load/unload path — it does not rely on the `AlignDevicesHook`s that `disk_offload` installs.

## Appendix D — Gotchas (ranked by cost)

From real integration work. Every one of these cost a nontrivial number of hours; document them so the next person doesn't pay the same tuition.

### D.1 safetensors dtype strings ≠ torch attribute names

`safe_open.get_slice(k).get_dtype()` returns `"BF16"`, but `OffloadedWeightsLoader.__getitem__` calls `getattr(torch, weight_info["dtype"])` which expects `"bfloat16"`. Need a translation table (see `_SAFE_TO_TORCH_DTYPE` in Appendix C).

Symptom: `AttributeError: module 'torch' has no attribute 'BF16'`.

### D.2 Tied weights are not in the safetensors

Llama's `lm_head.weight` is tied to `model.embed_tokens.weight`; only the latter is saved. Hooks look up by absolute module-parameter path, so both keys must be in the index, pointing at the same underlying data.

Detection requires `named_parameters(remove_duplicate=False)` — the default hides aliases. Also requires `model.tie_weights()` to have been called so identity-based detection works.

Symptom: `KeyError: 'lm_head.weight'`.

### D.3 Index timing is load-bearing

`OffloadedWeightsLoader` reads `index.json` once at construction and caches it. Writing the index **after** calling `disk_offload` silently has no effect.

Order: build index → alias tied weights → write to disk → call `disk_offload`.

### D.4 `execution_device` wants a `torch.device`, not a string

`"cuda:0"` works sometimes and fails with a cryptic type error elsewhere. `torch.device("cuda:0")` is safer.

### D.5 Pass 0 (embedding) is a special case

The residual buffer is populated by running the embedding layer over the whole dataset before the main loop starts. Embedding weights are much smaller than a transformer block — keep them resident on GPU for pass 0 rather than streaming them. Saves a load/unload per run.

### D.6 Attention mask shape assumptions

Some HF model implementations expect a 2D `attention_mask`; others broadcast a 4D causal mask internally. When you call `model.layers[i](x)` directly (bypassing the top-level forward), you may have to construct the mask yourself in the shape the layer expects. Model-family-specific; document per family in `fpwap.models.*`.
