#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checks import (
    check_ambiguous,
    check_false_premise,
    check_missing_capability,
    check_physically_infeasible,
    check_subjective,
    check_underspecified_locations,
    check_underspecified_objects,
)
from models import Phase2Checks, Phase2Output, parse_phase1_output, to_json_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 constraint derivation.")
    parser.add_argument("--input", type=Path, required=True, help="Input phase1.json path")
    parser.add_argument("--output", type=Path, required=True, help="Output phase2.json path")
    return parser.parse_args()


def load_phase1_input(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input file is not valid JSON: {input_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Input JSON must be an object: {input_path}")
    try:
        return parse_phase1_output(data)
    except KeyError as exc:
        raise ValueError(f"Input JSON is missing required field: {exc.args[0]}") from exc
    except TypeError as exc:
        raise ValueError(f"Input JSON has malformed structure: {exc}") from exc


def main() -> int:
    args = parse_args()
    scene = load_phase1_input(args.input)

    ambiguous = check_ambiguous(scene)
    false_prem = check_false_premise(scene)
    infeasible = check_physically_infeasible(scene)
    missing_cap = check_missing_capability(scene)
    subjective = check_subjective(scene)
    underspecified_objects = check_underspecified_objects(scene)
    underspecified_locations = check_underspecified_locations(scene)

    phase2_output = Phase2Output(
        scene_type=scene.scene_type,
        scene_objects=scene.scene_objects,
        scene_locations=scene.scene_locations,
        absent_and_implausible_objects=scene.absent_and_implausible_objects,
        checks=Phase2Checks(
            ambiguous_candidates=ambiguous,
            false_premise_candidates=false_prem,
            physically_infeasible_pairs=infeasible,
            missing_capability_candidates=missing_cap,
            subjective_candidates=subjective,
            underspecified_object_candidates=underspecified_objects,
            underspecified_location_candidates=underspecified_locations,
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(to_json_dict(phase2_output), indent=4) + "\n",
        encoding="utf-8",
    )

    print(f"ambiguous_classes:             {len(ambiguous)} candidates")
    print(f"false_premise_candidates:      {len(false_prem)} candidates")
    print(f"physically_infeasible_pairs:   {len(infeasible)} candidates")
    print(f"missing_capability_candidates: {len(missing_cap)} candidates")
    print(f"subjective_resolvable_classes: {len(subjective)} candidates")
    print(f"underspecified_objects:        {len(underspecified_objects)} candidates")
    print(f"underspecified_locations:      {len(underspecified_locations)} candidates")
    print(f"Phase 2 complete -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
