# SPDX-License-Identifier: Apache-2.0
"""Validate runner-enforced CPU affinity and write its artifact evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tests.utils.ci_resource_contract import collect_cpu_resource_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = collect_cpu_resource_contract(require_partition=True)
    evidence["github"] = {
        "job": os.environ.get("GITHUB_JOB"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner_name": os.environ.get("RUNNER_NAME"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if evidence["valid"]:
        return 0
    for error in evidence["errors"]:
        print(f"::error::{error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
