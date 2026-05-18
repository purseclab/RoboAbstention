#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ambiguous_referent import generate_instructions as generate_ambiguous_referent_instructions
from missing_capability import generate_instructions as generate_missing_capability_instructions
from contradictory_instructions import generate_instructions as generate_contradictory_instructions
from false_premise import generate_instructions as generate_false_premise_instructions
from missing_referent import generate_instructions
from physical_infeasibility import generate_instructions as generate_physical_infeasibility_instructions
from subjective_intent import generate_instructions as generate_subjective_intent_instructions
from underspecified_intent import generate_instructions as generate_underspecified_intent_instructions
from pipeline.phase2.models import parse_phase2_output


DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output" / "phase3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 instruction generation.")
    parser.add_argument("--input", type=Path, required=True, help="Input phase2.json path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output phase3.json path. Defaults to output/phase3/<input-name>",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic grammar sampling. Default: 0",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return DEFAULT_OUTPUT_DIR / input_path.name


def load_phase2_input(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input file is not valid JSON: {input_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Input JSON must be an object: {input_path}")
    try:
        return parse_phase2_output(data)
    except KeyError as exc:
        raise ValueError(f"Input JSON is missing required field: {exc.args[0]}") from exc
    except TypeError as exc:
        raise ValueError(f"Input JSON has malformed structure: {exc}") from exc


def main() -> int:
    args = parse_args()
    scene = load_phase2_input(args.input)
    output_path = resolve_output_path(args.input, args.output)

    payload = {
        "missing_referent": generate_instructions(scene, seed=args.seed),
        "ambiguous_referent": generate_ambiguous_referent_instructions(scene, seed=args.seed),
        "physical_infeasibility": generate_physical_infeasibility_instructions(scene, seed=args.seed),
        "false_premise": generate_false_premise_instructions(scene, seed=args.seed),
        "underspecified_intent": generate_underspecified_intent_instructions(scene, seed=args.seed),
        "subjective_intent": generate_subjective_intent_instructions(scene, seed=args.seed),
        "contradictory_instructions": generate_contradictory_instructions(scene, seed=args.seed),
        "missing_capability": generate_missing_capability_instructions(scene, seed=args.seed),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")

    print(
        "missing_referent: "
        f"{len(payload['missing_referent'])} instructions"
    )
    print(f"ambiguous_referent: {len(payload['ambiguous_referent'])} instructions")
    print(
        "physical_infeasibility: "
        f"{len(payload['physical_infeasibility'])} instructions"
    )
    print(f"false_premise: {len(payload['false_premise'])} instructions")
    print(f"underspecified_intent: {len(payload['underspecified_intent'])} instructions")
    print(f"subjective_intent: {len(payload['subjective_intent'])} instructions")
    print(
        "contradictory_instructions: "
        f"{len(payload['contradictory_instructions'])} instructions"
    )
    print(f"missing_capability: {len(payload['missing_capability'])} instructions")
    print(f"Phase 3 complete -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
