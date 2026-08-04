#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module

EXPECTED_SHA256 = "ee5a6a87a13a271c0418e07ea76fd4cba" "5cbee1af2d7dab53890caacb9254e14"


def main() -> None:
    spec = gen_cutlass_fused_moe_sm90_module()

    assert spec.is_aot, f"fused_moe_90 is not AOT; expected {spec.aot_path}"
    assert spec.aot_path.exists(), f"AOT artifact not found: {spec.aot_path}"

    digest = hashlib.sha256(spec.aot_path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, f"Unexpected fused_moe_90 artifact: {digest}"

    module = spec.build_and_load()
    print("Loaded AOT module:", module)
    print("AOT path:", spec.aot_path)
    print("SHA-256:", digest)


if __name__ == "__main__":
    main()
