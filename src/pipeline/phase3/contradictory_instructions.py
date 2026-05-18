from __future__ import annotations

import random
from dataclasses import asdict

from pipeline.phase2.models import Phase2Output, SceneLocation, SceneObject
from pipeline.phase2.size_order import parse_size
from pipeline.phase3.helper import (
    GeneratedInstruction,
    GrammarRule,
    format_object_name,
    sample_one,
)


TARGET_ON_LOCATION_TYPES = {"surface", "shelf", "floor_region"}
TARGET_IN_LOCATION_TYPES = {"container", "drawer", "inside_container"}
CURRENT_ON_LOCATION_TYPES = {"surface", "shelf", "floor_region"}
CURRENT_IN_LOCATION_TYPES = {"container", "drawer", "inside_container"}
CURRENT_AT_LOCATION_TYPES = {"wall_region", "hanging_point"}

STATE_ACTION_PAIRS = {
    "closed": "open",
    "open": "close",
    "off": "turn on",
    "on": "turn off",
    "empty": "fill",
    "full": "empty",
}


GRAMMAR = """
<contradictory_instruction> ::= <contact_contradiction>
                              | <state_preservation_contradiction>
                              | <location_preservation_contradiction>

<contact_contradiction> ::= <pick_up_without_touching>
                          | <handover_without_holding>
                          | <move_without_interacting>
                          | <put_without_moving>

<state_preservation_contradiction> ::= <open_keep_closed>
                                     | <close_keep_open>
                                     | <turn_on_keep_off>
                                     | <turn_off_keep_on>
                                     | <fill_keep_empty>
                                     | <empty_keep_full>

<location_preservation_contradiction> ::= <put_target_keep_current>
                                        | <move_target_leave_current>
                                        | <bring_object_keep_current>
                                        | <place_target_without_moving_from_current>

<pick_up_without_touching> ::= "pick up the " <carryable_manipulable_object> " without touching it"
<handover_without_holding> ::= <handover_verb> " the " <carryable_manipulable_object> " without holding it"
<move_without_interacting> ::= "move the " <carryable_manipulable_object> " without interacting with it"
<put_without_moving> ::= "put the " <carryable_manipulable_object> " " <target_preposition> " the " <feasible_target_location> " without moving it"

<open_keep_closed> ::= "open the " <closed_stateful_object> " while keeping it closed"
<close_keep_open> ::= "close the " <open_stateful_object> " while keeping it open"
<turn_on_keep_off> ::= "turn on the " <off_stateful_object> " while keeping it off"
<turn_off_keep_on> ::= "turn off the " <on_stateful_object> " while keeping it on"
<fill_keep_empty> ::= "fill the " <empty_stateful_object> " while keeping it empty"
<empty_keep_full> ::= "empty the " <full_stateful_object> " while keeping it full"

<put_target_keep_current> ::= "put the " <carryable_manipulable_object> " " <target_preposition> " the " <feasible_target_location> " while keeping it " <current_preposition> " the " <current_location>
<move_target_leave_current> ::= "move the " <carryable_manipulable_object> " to the " <feasible_target_location> " while leaving it " <current_preposition> " the " <current_location>
<bring_object_keep_current> ::= "bring me the " <carryable_manipulable_object> " while keeping it " <current_preposition> " the " <current_location>
<place_target_without_moving_from_current> ::= "place the " <carryable_manipulable_object> " " <target_preposition> " the " <feasible_target_location> " without moving it from the " <current_location>

<handover_verb> ::= "hand me" | "give me" | "bring me"

<target_preposition> ::= "on" for target locations whose location_type is surface, shelf, or floor_region
                       | "in" for target locations whose location_type is container, drawer, or inside_container

<current_preposition> ::= "on" for current locations whose location_type is surface, shelf, or floor_region
                        | "in" for current locations whose location_type is container, drawer, or inside_container
                        | "at" for current locations whose location_type is wall_region or hanging_point

<carryable_manipulable_object> ::= a scene_object with is_manipulable=true and exceeds_weight_limit=false

<closed_stateful_object> ::= a scene_object with is_stateful=true and state="closed"
<open_stateful_object> ::= a scene_object with is_stateful=true and state="open"
<off_stateful_object> ::= a scene_object with is_stateful=true and state="off"
<on_stateful_object> ::= a scene_object with is_stateful=true and state="on"
<empty_stateful_object> ::= a scene_object with is_stateful=true and state="empty"
<full_stateful_object> ::= a scene_object with is_stateful=true and state="full"

<feasible_target_location> ::= a scene_location with location_type in target placement types, size >= selected object size, and id different from the selected object's current location_id
<current_location> ::= the scene_location referenced by the selected object's location_id

Target placement types:
- on: surface, shelf, floor_region
- in: container, drawer, inside_container
- exclude hanging_point as a target placement location

Current location types:
- on: surface, shelf, floor_region
- in: container, drawer, inside_container
- at: wall_region, hanging_point
""".strip()


RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        name="pick_up_without_touching",
        pattern="pick up the {object_class} without touching it",
        slots=("object_class",),
        notes="Requires a scene_object with is_manipulable=true and exceeds_weight_limit=false.",
    ),
    GrammarRule(
        name="handover_without_holding",
        pattern="{handover_verb} the {object_class} without holding it",
        slots=("handover_verb", "object_class"),
        notes="Requires a scene_object with is_manipulable=true and exceeds_weight_limit=false.",
    ),
    GrammarRule(
        name="move_without_interacting",
        pattern="move the {object_class} without interacting with it",
        slots=("object_class",),
        notes="Requires a scene_object with is_manipulable=true and exceeds_weight_limit=false.",
    ),
    GrammarRule(
        name="put_without_moving",
        pattern="put the {object_class} {target_preposition} the {target_location} without moving it",
        slots=("object_class", "target_preposition", "target_location"),
        notes=(
            "Requires a carryable manipulable object and a feasible target location. "
            "Use on for surface/shelf/floor_region and in for container/drawer/"
            "inside_container. Exclude hanging_point as a target."
        ),
    ),
    GrammarRule(
        name="open_keep_closed",
        pattern="open the {object_class} while keeping it closed",
        slots=("object_class",),
        notes='Requires a scene_object with is_stateful=true and state="closed".',
    ),
    GrammarRule(
        name="close_keep_open",
        pattern="close the {object_class} while keeping it open",
        slots=("object_class",),
        notes='Requires a scene_object with is_stateful=true and state="open".',
    ),
    GrammarRule(
        name="turn_on_keep_off",
        pattern="turn on the {object_class} while keeping it off",
        slots=("object_class",),
        notes='Requires a scene_object with is_stateful=true and state="off".',
    ),
    GrammarRule(
        name="turn_off_keep_on",
        pattern="turn off the {object_class} while keeping it on",
        slots=("object_class",),
        notes='Requires a scene_object with is_stateful=true and state="on".',
    ),
    GrammarRule(
        name="fill_keep_empty",
        pattern="fill the {object_class} while keeping it empty",
        slots=("object_class",),
        notes='Requires a scene_object with is_stateful=true and state="empty".',
    ),
    GrammarRule(
        name="empty_keep_full",
        pattern="empty the {object_class} while keeping it full",
        slots=("object_class",),
        notes='Requires a scene_object with is_stateful=true and state="full".',
    ),
    GrammarRule(
        name="put_target_keep_current",
        pattern=(
            "put the {object_class} {target_preposition} the {target_location} "
            "while keeping it {current_preposition} the {current_location}"
        ),
        slots=(
            "object_class",
            "target_preposition",
            "target_location",
            "current_preposition",
            "current_location",
        ),
        notes=(
            "Requires a carryable manipulable object with a valid current location "
            "and a different feasible target placement location."
        ),
    ),
    GrammarRule(
        name="move_target_leave_current",
        pattern=(
            "move the {object_class} to the {target_location} "
            "while leaving it {current_preposition} the {current_location}"
        ),
        slots=("object_class", "target_location", "current_preposition", "current_location"),
        notes=(
            "Requires a carryable manipulable object with a valid current location "
            "and a different feasible target location."
        ),
    ),
    GrammarRule(
        name="bring_object_keep_current",
        pattern="bring me the {object_class} while keeping it {current_preposition} the {current_location}",
        slots=("object_class", "current_preposition", "current_location"),
        notes="Requires a carryable manipulable object with a valid current location.",
    ),
    GrammarRule(
        name="place_target_without_moving_from_current",
        pattern=(
            "place the {object_class} {target_preposition} the {target_location} "
            "without moving it from the {current_location}"
        ),
        slots=("object_class", "target_preposition", "target_location", "current_location"),
        notes=(
            "Requires a carryable manipulable object with a valid current location "
            "and a different feasible target placement location."
        ),
    ),
)


def get_grammar() -> str:
    return GRAMMAR


def get_rules() -> tuple[GrammarRule, ...]:
    return RULES


HANDOVER_VERBS = ("hand me", "give me", "bring me")


def _carryable_objects(scene: Phase2Output) -> list[SceneObject]:
    return [
        obj
        for obj in scene.scene_objects
        if obj.is_manipulable and not obj.exceeds_weight_limit
    ]


def _location_by_id(scene: Phase2Output) -> dict[str, SceneLocation]:
    return {location.id: location for location in scene.scene_locations}


def _target_preposition(location: SceneLocation) -> str | None:
    if location.location_type in TARGET_ON_LOCATION_TYPES:
        return "on"
    if location.location_type in TARGET_IN_LOCATION_TYPES:
        return "in"
    return None


def _current_preposition(location: SceneLocation) -> str | None:
    if location.location_type in CURRENT_ON_LOCATION_TYPES:
        return "on"
    if location.location_type in CURRENT_IN_LOCATION_TYPES:
        return "in"
    if location.location_type in CURRENT_AT_LOCATION_TYPES:
        return "at"
    return None


def _sample_feasible_target_location(
    scene: Phase2Output,
    obj: SceneObject,
    rng: random.Random,
    *,
    exclude_current: bool,
) -> tuple[SceneLocation, str] | None:
    candidates = list(scene.scene_locations)
    rng.shuffle(candidates)
    for location in candidates:
        if exclude_current and location.id == obj.location_id:
            continue
        preposition = _target_preposition(location)
        if preposition is None:
            continue
        if parse_size(location.size) < parse_size(obj.size):
            continue
        return location, preposition
    return None


def _sample_carryable_object_with_target(
    scene: Phase2Output,
    rng: random.Random,
    *,
    exclude_current: bool,
) -> tuple[SceneObject, SceneLocation, str] | None:
    objects = _carryable_objects(scene)
    rng.shuffle(objects)
    for obj in objects:
        location_with_preposition = _sample_feasible_target_location(
            scene,
            obj,
            rng,
            exclude_current=exclude_current,
        )
        if location_with_preposition is None:
            continue
        location, preposition = location_with_preposition
        return obj, location, preposition
    return None


def _sample_object_with_current_and_target(
    scene: Phase2Output,
    rng: random.Random,
) -> tuple[SceneObject, SceneLocation, str, SceneLocation, str] | None:
    locations = _location_by_id(scene)
    objects = _carryable_objects(scene)
    rng.shuffle(objects)
    for obj in objects:
        current_location = locations.get(obj.location_id)
        if current_location is None:
            continue
        current_preposition = _current_preposition(current_location)
        if current_preposition is None:
            continue
        target = _sample_feasible_target_location(
            scene,
            obj,
            rng,
            exclude_current=True,
        )
        if target is None:
            continue
        target_location, target_preposition = target
        return obj, target_location, target_preposition, current_location, current_preposition
    return None


def _stateful_objects(scene: Phase2Output, state: str) -> list[SceneObject]:
    return [
        obj
        for obj in scene.scene_objects
        if obj.is_stateful and obj.state == state
    ]


def _generate_pick_up_without_touching(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    obj = sample_one(_carryable_objects(scene), rng)
    if obj is None:
        return None
    return GeneratedInstruction(
        instruction=f"pick up the {format_object_name(obj.object_class)} without touching it",
        grammar_rule="pick_up_without_touching",
    )


def _generate_handover_without_holding(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    obj = sample_one(_carryable_objects(scene), rng)
    if obj is None:
        return None
    verb = sample_one(list(HANDOVER_VERBS), rng)
    return GeneratedInstruction(
        instruction=(
            f"{verb} the {format_object_name(obj.object_class)} without holding it"
        ),
        grammar_rule="handover_without_holding",
    )


def _generate_move_without_interacting(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    obj = sample_one(_carryable_objects(scene), rng)
    if obj is None:
        return None
    return GeneratedInstruction(
        instruction=f"move the {format_object_name(obj.object_class)} without interacting with it",
        grammar_rule="move_without_interacting",
    )


def _generate_put_without_moving(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    selection = _sample_carryable_object_with_target(scene, rng, exclude_current=True)
    if selection is None:
        return None
    obj, target_location, preposition = selection
    return GeneratedInstruction(
        instruction=(
            f"put the {format_object_name(obj.object_class)} {preposition} "
            f"the {target_location.description} without moving it"
        ),
        grammar_rule="put_without_moving",
    )


def _generate_state_preservation(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    state: str,
    action: str,
) -> GeneratedInstruction | None:
    obj = sample_one(_stateful_objects(scene, state), rng)
    if obj is None:
        return None
    return GeneratedInstruction(
        instruction=(
            f"{action} the {format_object_name(obj.object_class)} while keeping it {state}"
        ),
        grammar_rule=grammar_rule,
    )


def _generate_open_keep_closed(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_preservation(
        scene, rng, grammar_rule="open_keep_closed", state="closed", action="open"
    )


def _generate_close_keep_open(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_preservation(
        scene, rng, grammar_rule="close_keep_open", state="open", action="close"
    )


def _generate_turn_on_keep_off(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_preservation(
        scene, rng, grammar_rule="turn_on_keep_off", state="off", action="turn on"
    )


def _generate_turn_off_keep_on(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_preservation(
        scene, rng, grammar_rule="turn_off_keep_on", state="on", action="turn off"
    )


def _generate_fill_keep_empty(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_preservation(
        scene, rng, grammar_rule="fill_keep_empty", state="empty", action="fill"
    )


def _generate_empty_keep_full(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_preservation(
        scene, rng, grammar_rule="empty_keep_full", state="full", action="empty"
    )


def _generate_put_target_keep_current(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    selection = _sample_object_with_current_and_target(scene, rng)
    if selection is None:
        return None
    obj, target_location, target_preposition, current_location, current_preposition = selection
    return GeneratedInstruction(
        instruction=(
            f"put the {format_object_name(obj.object_class)} {target_preposition} "
            f"the {target_location.description} while keeping it "
            f"{current_preposition} the {current_location.description}"
        ),
        grammar_rule="put_target_keep_current",
    )


def _generate_move_target_leave_current(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    selection = _sample_object_with_current_and_target(scene, rng)
    if selection is None:
        return None
    obj, target_location, _, current_location, current_preposition = selection
    return GeneratedInstruction(
        instruction=(
            f"move the {format_object_name(obj.object_class)} to the "
            f"{target_location.description} while leaving it {current_preposition} "
            f"the {current_location.description}"
        ),
        grammar_rule="move_target_leave_current",
    )


def _generate_bring_object_keep_current(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    locations = _location_by_id(scene)
    candidates = _carryable_objects(scene)
    rng.shuffle(candidates)
    for obj in candidates:
        current_location = locations.get(obj.location_id)
        if current_location is None:
            continue
        current_preposition = _current_preposition(current_location)
        if current_preposition is None:
            continue
        return GeneratedInstruction(
            instruction=(
                f"bring me the {format_object_name(obj.object_class)} while keeping it "
                f"{current_preposition} the {current_location.description}"
            ),
            grammar_rule="bring_object_keep_current",
        )
    return None


def _generate_place_target_without_moving_from_current(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    selection = _sample_object_with_current_and_target(scene, rng)
    if selection is None:
        return None
    obj, target_location, target_preposition, current_location, _ = selection
    return GeneratedInstruction(
        instruction=(
            f"place the {format_object_name(obj.object_class)} {target_preposition} "
            f"the {target_location.description} without moving it from the "
            f"{current_location.description}"
        ),
        grammar_rule="place_target_without_moving_from_current",
    )


RULE_GENERATORS = {
    "pick_up_without_touching": _generate_pick_up_without_touching,
    "handover_without_holding": _generate_handover_without_holding,
    "move_without_interacting": _generate_move_without_interacting,
    "put_without_moving": _generate_put_without_moving,
    "open_keep_closed": _generate_open_keep_closed,
    "close_keep_open": _generate_close_keep_open,
    "turn_on_keep_off": _generate_turn_on_keep_off,
    "turn_off_keep_on": _generate_turn_off_keep_on,
    "fill_keep_empty": _generate_fill_keep_empty,
    "empty_keep_full": _generate_empty_keep_full,
    "put_target_keep_current": _generate_put_target_keep_current,
    "move_target_leave_current": _generate_move_target_leave_current,
    "bring_object_keep_current": _generate_bring_object_keep_current,
    "place_target_without_moving_from_current": _generate_place_target_without_moving_from_current,
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
