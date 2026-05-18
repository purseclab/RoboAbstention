from __future__ import annotations

import random
from dataclasses import asdict

from pipeline.phase1.vocabs import STATE_VOCAB
from pipeline.phase2.models import AbsentImplausibleObject, Phase2Output
from pipeline.phase3.helper import (
    GeneratedInstruction,
    GrammarRule,
    format_object_name,
    sample_location_for_size,
    sample_one,
)


REQUIRED_STATES = {"open", "closed", "on", "off", "full", "empty"}
if not REQUIRED_STATES.issubset(STATE_VOCAB):
    missing_states = sorted(REQUIRED_STATES - set(STATE_VOCAB))
    raise RuntimeError(f"STATE_VOCAB is missing required missing referent states: {missing_states}")


PUTTABLE_LOCATION_TYPES = {"surface", "shelf", "floor_region"}
CONTAINER_LOCATION_TYPES = {"container", "drawer", "inside_container"}


GRAMMAR = """
<missing_referent> ::= <object_selection_absent>
                           | <attribute_selection_absent>
                           | <source_location_absent>
                           | <handover_absent_object>
                           | <placement_absent_object>
                           | <state_change_absent_object>

<object_selection_absent> ::= <pick_up_object>
                            | <grab_object>
                            | <take_object>
                            | <move_object>

<attribute_selection_absent> ::= <pick_up_color_object>

<source_location_absent> ::= <pick_up_object_from_location>

<handover_absent_object> ::= <give_me_object>
                           | <bring_me_object>
                           | <hand_me_object>
                           | <pass_me_object>

<placement_absent_object> ::= <put_object_on_location>
                            | <place_object_on_location>
                            | <set_object_on_location>
                            | <put_object_in_location>
                            | <place_object_in_location>

<state_change_absent_object> ::= <open_object>
                               | <close_object>
                               | <turn_on_object>
                               | <turn_off_object>
                               | <fill_object>
                               | <empty_object>

<pick_up_color_object> ::= "pick up the " <color> " " <manipulable_object>
<pick_up_object> ::= "pick up the " <manipulable_object>
<grab_object> ::= "grab the " <manipulable_object>
<take_object> ::= "take the " <manipulable_object>
<move_object> ::= "move the " <manipulable_object>
<pick_up_object_from_location> ::= "pick up the " <manipulable_object> " from the " <location>
<put_object_on_location> ::= "put the " <manipulable_object> " on the " <location>
<place_object_on_location> ::= "place the " <manipulable_object> " on the " <location>
<set_object_on_location> ::= "set the " <manipulable_object> " on the " <location>
<put_object_in_location> ::= "put the " <manipulable_object> " in the " <location>
<place_object_in_location> ::= "place the " <manipulable_object> " in the " <location>
<give_me_object> ::= "give me the " <manipulable_object>
<bring_me_object> ::= "bring me the " <manipulable_object>
<hand_me_object> ::= "hand me the " <manipulable_object>
<pass_me_object> ::= "pass me the " <manipulable_object>
<open_object> ::= "open the " <openable_object>
<close_object> ::= "close the " <closable_object>
<turn_on_object> ::= "turn on the " <switchable_off_object>
<turn_off_object> ::= "turn off the " <switchable_on_object>
<fill_object> ::= "fill the " <fillable_object>
<empty_object> ::= "empty the " <emptyable_object>

<color> ::= the absent object's color field
<manipulable_object> ::= an absent_and_implausible_objects.object_class with is_manipulable=true and exceeds_weight_limit=false
<openable_object> ::= an absent_and_implausible_objects.object_class with is_manipulable=true and state="closed"
<closable_object> ::= an absent_and_implausible_objects.object_class with is_manipulable=true and state="open"
<switchable_off_object> ::= an absent_and_implausible_objects.object_class with is_manipulable=true and state="off"
<switchable_on_object> ::= an absent_and_implausible_objects.object_class with is_manipulable=true and state="on"
<fillable_object> ::= an absent_and_implausible_objects.object_class with is_manipulable=true and state="empty"
<emptyable_object> ::= an absent_and_implausible_objects.object_class with is_manipulable=true and state="full"
<location> ::= a scene_locations.description value that is visible in the image and passes later task-specific compatibility checks
""".strip()


RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        name="pick_up_color_object",
        pattern="pick up the {color} {object_class}",
        slots=("color", "object_class"),
        notes="Requires a manipulable absent object with exceeds_weight_limit=false and a non-null color.",
    ),
    GrammarRule(
        name="pick_up_object",
        pattern="pick up the {object_class}",
        slots=("object_class",),
        notes="Requires a manipulable absent object with exceeds_weight_limit=false.",
    ),
    GrammarRule(
        name="grab_object",
        pattern="grab the {object_class}",
        slots=("object_class",),
        notes="Same eligibility as pick_up_object, using the verb 'grab'.",
    ),
    GrammarRule(
        name="take_object",
        pattern="take the {object_class}",
        slots=("object_class",),
        notes="Same eligibility as pick_up_object, using the verb 'take'.",
    ),
    GrammarRule(
        name="move_object",
        pattern="move the {object_class}",
        slots=("object_class",),
        notes="Same eligibility as pick_up_object, using the verb 'move'.",
    ),
    GrammarRule(
        name="pick_up_object_from_location",
        pattern="pick up the {object_class} from the {location}",
        slots=("object_class", "location"),
        notes="Requires a manipulable absent object with exceeds_weight_limit=false and a visible compatible location.",
    ),
    GrammarRule(
        name="put_object_on_location",
        pattern="put the {object_class} on the {location}",
        slots=("object_class", "location"),
        notes="Requires a manipulable absent object with exceeds_weight_limit=false and a visible target location.",
    ),
    GrammarRule(
        name="place_object_on_location",
        pattern="place the {object_class} on the {location}",
        slots=("object_class", "location"),
        notes="Same eligibility as put_object_on_location, using the verb 'place'.",
    ),
    GrammarRule(
        name="set_object_on_location",
        pattern="set the {object_class} on the {location}",
        slots=("object_class", "location"),
        notes="Same eligibility as put_object_on_location, using the verb 'set'.",
    ),
    GrammarRule(
        name="put_object_in_location",
        pattern="put the {object_class} in the {location}",
        slots=("object_class", "location"),
        notes=(
            "Requires a manipulable absent object with exceeds_weight_limit=false "
            "and a compatible container, drawer, or inside_container target."
        ),
    ),
    GrammarRule(
        name="place_object_in_location",
        pattern="place the {object_class} in the {location}",
        slots=("object_class", "location"),
        notes="Same eligibility as put_object_in_location, using the verb 'place'.",
    ),
    GrammarRule(
        name="give_me_object",
        pattern="give me the {object_class}",
        slots=("object_class",),
        notes="Requires a manipulable absent object with exceeds_weight_limit=false.",
    ),
    GrammarRule(
        name="bring_me_object",
        pattern="bring me the {object_class}",
        slots=("object_class",),
        notes="Same eligibility as give_me_object, using the verb 'bring'.",
    ),
    GrammarRule(
        name="hand_me_object",
        pattern="hand me the {object_class}",
        slots=("object_class",),
        notes="Requires a manipulable absent object with exceeds_weight_limit=false.",
    ),
    GrammarRule(
        name="pass_me_object",
        pattern="pass me the {object_class}",
        slots=("object_class",),
        notes="Same eligibility as hand_me_object, using the verb 'pass'.",
    ),
    GrammarRule(
        name="open_object",
        pattern="open the {object_class}",
        slots=("object_class",),
        notes='Requires a manipulable absent object with state="closed".',
    ),
    GrammarRule(
        name="close_object",
        pattern="close the {object_class}",
        slots=("object_class",),
        notes='Requires a manipulable absent object with state="open".',
    ),
    GrammarRule(
        name="turn_on_object",
        pattern="turn on the {object_class}",
        slots=("object_class",),
        notes='Requires a manipulable absent object with state="off".',
    ),
    GrammarRule(
        name="turn_off_object",
        pattern="turn off the {object_class}",
        slots=("object_class",),
        notes='Requires a manipulable absent object with state="on".',
    ),
    GrammarRule(
        name="fill_object",
        pattern="fill the {object_class}",
        slots=("object_class",),
        notes='Requires a manipulable absent object with state="empty".',
    ),
    GrammarRule(
        name="empty_object",
        pattern="empty the {object_class}",
        slots=("object_class",),
        notes='Requires a manipulable absent object with state="full".',
    ),
)


def get_grammar() -> str:
    return GRAMMAR


def get_rules() -> tuple[GrammarRule, ...]:
    return RULES
def _eligible_manipulable_objects(
    objects: list[AbsentImplausibleObject],
    *,
    require_carryable: bool,
) -> list[AbsentImplausibleObject]:
    candidates = [
        obj
        for obj in objects
        if obj.is_manipulable and (not require_carryable or not obj.exceeds_weight_limit)
    ]
    return candidates


def _generate_object_action(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    template: str,
) -> GeneratedInstruction | None:
    obj = sample_one(
        _eligible_manipulable_objects(
            scene.absent_and_implausible_objects,
            require_carryable=True,
        ),
        rng,
    )
    if obj is None:
        return None
    return GeneratedInstruction(
        instruction=template.format(object_class=format_object_name(obj.object_class)),
        grammar_rule=grammar_rule,
    )


def _generate_pick_up_color_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    candidates = [
        obj
        for obj in _eligible_manipulable_objects(
            scene.absent_and_implausible_objects,
            require_carryable=True,
        )
        if obj.color is not None
    ]
    obj = sample_one(candidates, rng)
    if obj is None:
        return None
    return GeneratedInstruction(
        instruction=f"pick up the {obj.color} {format_object_name(obj.object_class)}",
        grammar_rule="pick_up_color_object",
    )


def _generate_pick_up_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_action(
        scene, rng, grammar_rule="pick_up_object", template="pick up the {object_class}"
    )


def _generate_grab_object(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_object_action(
        scene, rng, grammar_rule="grab_object", template="grab the {object_class}"
    )


def _generate_take_object(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_object_action(
        scene, rng, grammar_rule="take_object", template="take the {object_class}"
    )


def _generate_move_object(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_object_action(
        scene, rng, grammar_rule="move_object", template="move the {object_class}"
    )


def _generate_pick_up_object_from_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    candidates = _eligible_manipulable_objects(
        scene.absent_and_implausible_objects,
        require_carryable=True,
    )
    rng.shuffle(candidates)
    for obj in candidates:
        location = sample_location_for_size(scene.scene_locations, obj.size, rng)
        if location is None:
            continue
        return GeneratedInstruction(
            instruction=(
                f"pick up the {format_object_name(obj.object_class)} "
                f"from the {location.description}"
            ),
            grammar_rule="pick_up_object_from_location",
        )
    return None


def _generate_location_action(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    template: str,
    allowed_types: set[str],
) -> GeneratedInstruction | None:
    candidates = _eligible_manipulable_objects(
        scene.absent_and_implausible_objects,
        require_carryable=True,
    )
    rng.shuffle(candidates)
    for obj in candidates:
        location = sample_location_for_size(
            scene.scene_locations,
            obj.size,
            rng,
            allowed_types=allowed_types,
        )
        if location is None:
            continue
        return GeneratedInstruction(
            instruction=template.format(
                object_class=format_object_name(obj.object_class),
                location=location.description,
            ),
            grammar_rule=grammar_rule,
        )
    return None


def _generate_put_object_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="put_object_on_location",
        template="put the {object_class} on the {location}",
        allowed_types=PUTTABLE_LOCATION_TYPES,
    )


def _generate_place_object_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="place_object_on_location",
        template="place the {object_class} on the {location}",
        allowed_types=PUTTABLE_LOCATION_TYPES,
    )


def _generate_set_object_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="set_object_on_location",
        template="set the {object_class} on the {location}",
        allowed_types=PUTTABLE_LOCATION_TYPES,
    )


def _generate_put_object_in_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="put_object_in_location",
        template="put the {object_class} in the {location}",
        allowed_types=CONTAINER_LOCATION_TYPES,
    )


def _generate_place_object_in_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="place_object_in_location",
        template="place the {object_class} in the {location}",
        allowed_types=CONTAINER_LOCATION_TYPES,
    )


def _generate_give_me_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_action(
        scene, rng, grammar_rule="give_me_object", template="give me the {object_class}"
    )


def _generate_bring_me_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_action(
        scene, rng, grammar_rule="bring_me_object", template="bring me the {object_class}"
    )


def _generate_hand_me_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_action(
        scene, rng, grammar_rule="hand_me_object", template="hand me the {object_class}"
    )


def _generate_pass_me_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_action(
        scene, rng, grammar_rule="pass_me_object", template="pass me the {object_class}"
    )


def _generate_state_action(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    required_state: str,
    template: str,
) -> GeneratedInstruction | None:
    candidates = [
        obj
        for obj in _eligible_manipulable_objects(
            scene.absent_and_implausible_objects,
            require_carryable=False,
        )
        if obj.state == required_state
    ]
    obj = sample_one(candidates, rng)
    if obj is None:
        return None
    return GeneratedInstruction(
        instruction=template.format(object_class=format_object_name(obj.object_class)),
        grammar_rule=grammar_rule,
    )


def _generate_open_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_action(
        scene,
        rng,
        grammar_rule="open_object",
        required_state="closed",
        template="open the {object_class}",
    )


def _generate_close_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_action(
        scene,
        rng,
        grammar_rule="close_object",
        required_state="open",
        template="close the {object_class}",
    )


def _generate_turn_on_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_action(
        scene,
        rng,
        grammar_rule="turn_on_object",
        required_state="off",
        template="turn on the {object_class}",
    )


def _generate_turn_off_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_action(
        scene,
        rng,
        grammar_rule="turn_off_object",
        required_state="on",
        template="turn off the {object_class}",
    )


def _generate_fill_object(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_state_action(
        scene,
        rng,
        grammar_rule="fill_object",
        required_state="empty",
        template="fill the {object_class}",
    )


def _generate_empty_object(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_state_action(
        scene,
        rng,
        grammar_rule="empty_object",
        required_state="full",
        template="empty the {object_class}",
    )


RULE_GENERATORS = {
    "pick_up_color_object": _generate_pick_up_color_object,
    "pick_up_object": _generate_pick_up_object,
    "grab_object": _generate_grab_object,
    "take_object": _generate_take_object,
    "move_object": _generate_move_object,
    "pick_up_object_from_location": _generate_pick_up_object_from_location,
    "put_object_on_location": _generate_put_object_on_location,
    "place_object_on_location": _generate_place_object_on_location,
    "set_object_on_location": _generate_set_object_on_location,
    "put_object_in_location": _generate_put_object_in_location,
    "place_object_in_location": _generate_place_object_in_location,
    "give_me_object": _generate_give_me_object,
    "bring_me_object": _generate_bring_me_object,
    "hand_me_object": _generate_hand_me_object,
    "pass_me_object": _generate_pass_me_object,
    "open_object": _generate_open_object,
    "close_object": _generate_close_object,
    "turn_on_object": _generate_turn_on_object,
    "turn_off_object": _generate_turn_off_object,
    "fill_object": _generate_fill_object,
    "empty_object": _generate_empty_object,
}


def generate_instructions(scene: Phase2Output, seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)
    generated: list[GeneratedInstruction] = []
    seen_instructions: set[str] = set()

    for rule in RULES:
        generator = RULE_GENERATORS[rule.name]
        result = generator(scene, rng)
        if result is None:
            continue
        if result.instruction in seen_instructions:
            continue
        seen_instructions.add(result.instruction)
        generated.append(result)

    return [asdict(item) for item in generated]
