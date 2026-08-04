"""Single-module AOT root for Omni's validated SM90 fused-MoE artifact."""

from pathlib import Path

__version__ = "0.6.14+cu130.omni1"


def get_jit_cache_dir() -> str:
    return str(Path(__file__).parent / "data")
