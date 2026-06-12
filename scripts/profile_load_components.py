"""Component profile of one layer's load path on a real snapshot.

Times, for one MoE layer:
  1. loader[name] for every param (safetensors get_tensor / conversion)
  2. copy of those tensors into a pinned buffer
  3. one async H2D from pinned
  4. legacy per-tensor set_module_tensor_to_device path
Run twice back-to-back: pass 2 shows the page-cache-warm numbers.

Usage:
    uv run scripts/profile_load_components.py --snapshot <dir> --layer 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fpwap.engine import _OffloadStreamer
from fpwap.loader import build_empty_model_and_index
from fpwap.models import get_plumbing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--layer", type=int, default=10)
    args = ap.parse_args()
    snap = Path(args.snapshot).expanduser()

    model, accel_index, _ = build_empty_model_and_index(str(snap), snap)
    model.eval()
    plumbing = get_plumbing(model)
    streamer = _OffloadStreamer(accel_index, torch.device("cuda"), model=model)
    loader = streamer._loader
    layer = plumbing.layer_modules(model)[args.layer]
    prefix = plumbing.layer_prefix(args.layer)

    names = [rel for rel, _ in layer.named_parameters()]
    print(f"layer {args.layer}: {len(names)} params")

    for pass_idx in (1, 2):
        t0 = time.perf_counter()
        tensors = {rel: loader[f"{prefix}.{rel}"] for rel in names}
        t_read = time.perf_counter() - t0
        total = sum(t.element_size() * t.numel() for t in tensors.values())
        gb = total / 1e9

        pinned = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        t0 = time.perf_counter()
        off = 0
        for rel in names:
            t = tensors[rel]
            n = t.element_size() * t.numel()
            pinned[off : off + n].view(t.dtype).view(t.shape).copy_(t)
            off += n
        t_fill = time.perf_counter() - t0

        dev = torch.empty(total, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dev.copy_(pinned, non_blocking=True)
        torch.cuda.synchronize()
        t_h2d = time.perf_counter() - t0

        print(
            f"pass {pass_idx}: {gb:.2f} GB | "
            f"read {t_read:.3f}s ({gb / t_read:.1f} GB/s) | "
            f"pin-fill {t_fill:.3f}s ({gb / t_fill:.1f} GB/s) | "
            f"H2D {t_h2d:.3f}s ({gb / t_h2d:.1f} GB/s)"
        )
        del tensors, pinned, dev

    # Legacy path for reference (loads layer onto cuda, per-tensor).
    from fpwap.loader import _load_layer, _unload_layer

    for pass_idx in (1, 2):
        _unload_layer(model, args.layer, plumbing)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _load_layer(model, args.layer, plumbing, loader, torch.device("cuda"))
        torch.cuda.synchronize()
        t_legacy = time.perf_counter() - t0
        print(f"legacy per-tensor load pass {pass_idx}: {t_legacy:.3f}s")
    streamer.close()


if __name__ == "__main__":
    main()
