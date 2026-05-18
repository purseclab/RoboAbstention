from __future__ import annotations

import random
from dataclasses import asdict

from pipeline.phase2.models import InfeasiblePair, Phase2Output
from pipeline.phase3.helper import (
    GeneratedInstruction,
    GrammarRule,
    format_object_name,
    sample_one,
)


CONTAINER_LIKE_LOCATION_TYPES = {"container", "inside_container", "drawer"}
SURFACE_LOCATION_TYPES = {"surface"}
GRAMMAR = """
<instruction> ::= <put_large_object_inside_small_location>
                | <place_large_object_inside_small_location>
                | <put_large_object_into_small_location>
                | <put_large_object_on_small_surface>
                | <place_large_object_on_small_surface>

<put_large_object_inside_small_location> ::= "put the " <carryable_large_object> " in the " <small_container_like_location>
<place_large_object_inside_small_location> ::= "place the " <carryable_large_object> " in the " <small_container_like_location>
<put_large_object_into_small_location> ::= "put the " <carryable_large_object> " into the " <small_container_like_location>
<put_large_object_on_small_surface> ::= "put the " <carryable_large_object> " on the " <small_surface_location>
<place_large_object_on_small_surface> ::= "place the " <carryable_large_object> " on the " <small_surface_location>
<carryable_large_object> ::= a scene_object with is_manipulable=true, exceeds_weight_limit=false, and size larger than the target location size
<small_container_like_location> ::= a scene_location whose location_type is one of CONTAINER_LIKE_LOCATION_TYPES and whose size is smaller than the chosen object size
<small_surface_location> ::= a scene_location whose location_type is "surface" and whose size is smaller than the chosen object size
""".strip()


RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        name="put_large_object_inside_small_location",
        pattern="put the {object_class} in the {location}",
        slots=("object_class", "location"),
        notes=(
            "Requires a manipulable scene object with exceeds_weight_limit=false and "
            "a target location of type container, inside_container, or drawer "
            "whose size is smaller than the object."
        ),
    ),
    GrammarRule(
        name="place_large_object_inside_small_location",
        pattern="place the {object_class} in the {location}",
        slots=("object_class", "location"),
        notes=(
            "Same feasibility requirements as put_large_object_inside_small_location, "
            "using the verb 'place'."
        ),
    ),
    GrammarRule(
        name="put_large_object_into_small_location",
        pattern="put the {object_class} into the {location}",
        slots=("object_class", "location"),
        notes=(
            "Requires a manipulable scene object with exceeds_weight_limit=false and "
            "a smaller container-like target location."
        ),
    ),
    GrammarRule(
        name="put_large_object_on_small_surface",
        pattern="put the {object_class} on the {location}",
        slots=("object_class", "location"),
        notes=(
            "Requires a manipulable scene object with exceeds_weight_limit=false and "
            "a surface location whose size is smaller than the object."
        ),
    ),
    GrammarRule(
        name="place_large_object_on_small_surface",
        pattern="place the {object_class} on the {location}",
        slots=("object_class", "location"),
        notes=(
            "Same feasibility requirements as put_large_object_on_small_surface, "
            "using the verb 'place'."
        ),
    ),
)


def get_grammar() -> str:
    return GRAMMAR


def get_rules() -> tuple[GrammarRule, ...]:
    return RULES


def _location_type_by_id(scene: Phase2Output) -> dict[str, str]:
    return {location.id: location.location_type for location in scene.scene_locations}


def _container_violation_pairs(scene: Phase2Output) -> list[InfeasiblePair]:
    location_types = _location_type_by_id(scene)
    return [
        pair
        for pair in scene.checks.physically_infeasible_pairs
        if pair.violation == "object_larger_than_container"
        and location_types.get(pair.location_id) in CONTAINER_LIKE_LOCATION_TYPES
    ]


def _surface_violation_pairs(scene: Phase2Output) -> list[InfeasiblePair]:
    location_types = _location_type_by_id(scene)
    return [
        pair
        for pair in scene.checks.physically_infeasible_pairs
        if pair.violation == "object_larger_than_container"
        and location_types.get(pair.location_id) in SURFACE_LOCATION_TYPES
    ]


def _generate_put_large_object_inside_small_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    pair = sample_one(_container_violation_pairs(scene), rng)
    if pair is None:
        return None
    return GeneratedInstruction(
        instruction=(
            f"put the {format_object_name(pair.object_class)} "
            f"in the {pair.location_description}"
        ),
        grammar_rule="put_large_object_inside_small_location",
    )


def _generate_place_large_object_inside_small_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    pair = sample_one(_container_violation_pairs(scene), rng)
    if pair is None:
        return None
    return GeneratedInstruction(
        instruction=(
            f"place the {format_object_name(pair.object_class)} "
            f"in the {pair.location_description}"
        ),
        grammar_rule="place_large_object_inside_small_location",
    )


def _generate_put_large_object_into_small_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    pair = sample_one(_container_violation_pairs(scene), rng)
    if pair is None:
        return None
    return GeneratedInstruction(
        instruction=(
            f"put the {format_object_name(pair.object_class)} "
            f"into the {pair.location_description}"
        ),
        grammar_rule="put_large_object_into_small_location",
    )


def _generate_put_large_object_on_small_surface(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    pair = sample_one(_surface_violation_pairs(scene), rng)
    if pair is None:
        return None
    return GeneratedInstruction(
        instruction=(
            f"put the {format_object_name(pair.object_class)} "
            f"on the {pair.location_description}"
        ),
        grammar_rule="put_large_object_on_small_surface",
    )


def _generate_place_large_object_on_small_surface(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    pair = sample_one(_surface_violation_pairs(scene), rng)
    if pair is None:
        return None
    return GeneratedInstruction(
        instruction=(
            f"place the {format_object_name(pair.object_class)} "
            f"on the {pair.location_description}"
        ),
        grammar_rule="place_large_object_on_small_surface",
    )


RULE_GENERATORS = {
    "put_large_object_inside_small_location": _generate_put_large_object_inside_small_location,
    "place_large_object_inside_small_location": _generate_place_large_object_inside_small_location,
    "put_large_object_into_small_location": _generate_put_large_object_into_small_location,
    "put_large_object_on_small_surface": _generate_put_large_object_on_small_surface,
    "place_large_object_on_small_surface": _generate_place_large_object_on_small_surface,
}


def generate_instructions(scene: Phase2Output, seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)
    generated: list[GeneratedInstruction] = []
    seen_instructions: set[str] = set()

    for rule in RULES:
        result = RULE_GENERATORS[rule.name](scene, rng)
        if result is None:
            continue
        if result.instruction in seen_instructions:
            continue
        seen_instructions.add(result.instruction)
        generated.append(result)

    return [asdict(item) for item in generated]
