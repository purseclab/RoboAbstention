#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "output"
PHASE1_RUN = WORKSPACE_ROOT / "pipeline" / "phase1" / "run.py"
PHASE2_RUN = WORKSPACE_ROOT / "pipeline" / "phase2" / "run.py"
PHASE3_RUN = WORKSPACE_ROOT / "pipeline" / "phase3" / "run.py"
DEFAULT_MODEL = "openai/gpt-5.4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pipeline for one image.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Phase 1 LiteLLM model string. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Phase 3 random seed for deterministic grammar sampling. Default: 0",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for phase outputs. Default: output.",
    )
    return parser.parse_args()


def resolve_output_paths(
    image_path: Path, output_root: Path
) -> tuple[Path, Path, Path]:
    image_path = image_path.resolve()
    output_root = output_root.resolve()
    stem = image_path.stem
    return (
        output_root / "phase1" / f"{stem}.json",
        output_root / "phase2" / f"{stem}.json",
        output_root / "phase3" / f"{stem}.json",
    )


def run_step(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=WORKSPACE_ROOT)


def main() -> int:
    args = parse_args()
    (
        phase1_output,
        phase2_output,
        phase3_output,
    ) = resolve_output_paths(
        args.image,
        args.output_root,
    )
    phase1_command = [
        sys.executable,
        str(PHASE1_RUN),
        "--image",
        str(args.image),
        "--output",
        str(phase1_output),
        "--model",
        args.model,
    ]

    run_step(phase1_command)
    run_step(
        [
            sys.executable,
            str(PHASE2_RUN),
            "--input",
            str(phase1_output),
            "--output",
            str(phase2_output),
        ]
    )
    run_step(
        [
            sys.executable,
            str(PHASE3_RUN),
            "--input",
            str(phase2_output),
            "--output",
            str(phase3_output),
            "--seed",
            str(args.seed),
        ]
    )
    print(f"Phase 1 output -> {phase1_output}")
    print(f"Phase 2 output -> {phase2_output}")
    print(f"Phase 3 output -> {phase3_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
