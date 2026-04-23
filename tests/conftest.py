from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu-large",
        action="store_true",
        default=False,
        help="Run gpu_large tests (multi-GB real checkpoints, opt-in).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip @pytest.mark.gpu tests when no CUDA device is available.
    Skip @pytest.mark.gpu_large tests unless --run-gpu-large is passed."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    run_gpu_large = config.getoption("--run-gpu-large", default=False)

    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    skip_gpu_large = pytest.mark.skip(reason="needs --run-gpu-large flag")
    for item in items:
        if "gpu" in item.keywords and not cuda_available:
            item.add_marker(skip_gpu)
        if "gpu_large" in item.keywords and not run_gpu_large:
            item.add_marker(skip_gpu_large)
