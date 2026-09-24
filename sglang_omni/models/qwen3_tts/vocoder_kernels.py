# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the shared vocoder SnakeBeta implementation."""

from sglang_omni.utils.snake_beta import (
    FusedSnakeBeta,
    fuse_vocoder_decoder,
    fused_snake_beta,
)

__all__ = ["FusedSnakeBeta", "fuse_vocoder_decoder", "fused_snake_beta"]
