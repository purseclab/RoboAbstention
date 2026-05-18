from __future__ import annotations

import random
from dataclasses import dataclass

from pipeline.phase2.models import SceneLocation
from pipeline.phase2.size_order import parse_size


@dataclass(frozen=True)
class GrammarRule:
    name: str
    pattern: str
    slots: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class GeneratedInstruction:
    instruction: str
    grammar_rule: str


def format_object_name(object_class: str) -> str:
    return object_class.replace("_", " ")


def sample_one(items: list, rng: random.Random):
    if not items:
        return None
    return rng.choice(items)


def max_shared_size(sizes: list[str]) -> str | None:
    if not sizes:
        return None
    return max(sizes, key=parse_size)


def sample_location_for_size(
    locations: list[SceneLocation],
    size: str | None,
    rng: random.Random,
    *,
    allowed_types: set[str] | None = None,
) -> SceneLocation | None:
    candidates = []
    for location in locations:
        if allowed_types is not None and location.location_type not in allowed_types:
            continue
        if size is not None and parse_size(location.size) < parse_size(size):
            continue
        candidates.append(location)
    return sample_one(candidates, rng)
