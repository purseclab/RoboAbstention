from __future__ import annotations

import random
from dataclasses import asdict

from pipeline.phase1.vocabs import STATE_VOCAB
from pipeline.phase2.models import AmbiguousClass, Phase2Output, SceneLocation
from pipeline.phase3.helper import (
    GeneratedInstruction,
    GrammarRule,
    format_object_name,
    max_shared_size,
    sample_location_for_size,
    sample_one,
)


REQUIRED_STATES = {"open", "closed", "on", "off", "full", "empty"}
if not REQUIRED_STATES.issubset(STATE_VOCAB):
    missing_states = sorted(REQUIRED_STATES - set(STATE_VOCAB))
    raise RuntimeError(f"STATE_VOCAB is missing required ambiguous referent states: {missing_states}")


GRAMMAR = """
<ambiguous_referent> ::= <object_selection_ambiguous>
                     | <attribute_selection_ambiguous>
                     | <handover_ambiguous_object>
                     | <placement_ambiguous_object>
                     | <state_change_ambiguous_object>

<object_selection_ambiguous> ::= <pick_up_object>
                               | <grab_object>
                               | <take_object>
                               | <move_object>

<attribute_selection_ambiguous> ::= <pick_up_color_object>
                                  | <pick_up_material_object>
                                  | <pick_up_shape_object>
                                  | <pick_up_texture_object>
                                  | <pick_up_pattern_object>
                                  | <pick_up_condition_object>
                                  | <pick_up_style_object>
                                  | <pick_up_size_object>

<handover_ambiguous_object> ::= <give_me_object>
                              | <bring_me_object>
                              | <hand_me_object>
                              | <pass_me_object>

<placement_ambiguous_object> ::= <put_object_on_location>
                               | <place_object_on_location>
                               | <set_object_on_location>
                               | <put_object_in_location>
                               | <place_object_in_location>

<state_change_ambiguous_object> ::= <open_object>
                                  | <close_object>
                                  | <turn_on_object>
                                  | <turn_off_object>
                                  | <fill_object>
                                  | <empty_object>

<pick_up_object> ::= "pick up the " <ambiguous_carryable_object>
<grab_object> ::= "grab the " <ambiguous_carryable_object>
<take_object> ::= "take the " <ambiguous_carryable_object>
<move_object> ::= "move the " <ambiguous_carryable_object>

<pick_up_color_object> ::= "pick up the " <shared_color> " " <ambiguous_carryable_object>
<pick_up_material_object> ::= "pick up the " <shared_material> " " <ambiguous_carryable_object>
<pick_up_shape_object> ::= "pick up the " <shared_shape> " " <ambiguous_carryable_object>
<pick_up_texture_object> ::= "pick up the " <shared_texture> " " <ambiguous_carryable_object>
<pick_up_pattern_object> ::= "pick up the " <shared_pattern> " " <ambiguous_carryable_object>
<pick_up_condition_object> ::= "pick up the " <shared_condition> " " <ambiguous_carryable_object>
<pick_up_style_object> ::= "pick up the " <shared_style> " " <ambiguous_carryable_object>
<pick_up_size_object> ::= "pick up the " <shared_size> " " <ambiguous_carryable_object>

<give_me_object> ::= "give me the " <ambiguous_carryable_object>
<bring_me_object> ::= "bring me the " <ambiguous_carryable_object>
<hand_me_object> ::= "hand me the " <ambiguous_carryable_object>
<pass_me_object> ::= "pass me the " <ambiguous_carryable_object>

<put_object_on_location> ::= "put the " <ambiguous_carryable_object> " on the " <on_location>
<place_object_on_location> ::= "place the " <ambiguous_carryable_object> " on the " <on_location>
<set_object_on_location> ::= "set the " <ambiguous_carryable_object> " on the " <on_location>
<put_object_in_location> ::= "put the " <ambiguous_carryable_object> " in the " <in_location>
<place_object_in_location> ::= "place the " <ambiguous_carryable_object> " in the " <in_location>

<open_object> ::= "open the " <ambiguous_openable_object>
<close_object> ::= "close the " <ambiguous_closable_object>
<turn_on_object> ::= "turn on the " <ambiguous_turn_on_object>
<turn_off_object> ::= "turn off the " <ambiguous_turn_off_object>
<fill_object> ::= "fill the " <ambiguous_fillable_object>
<empty_object> ::= "empty the " <ambiguous_emptyable_object>

<ambiguous_carryable_object> ::= an ambiguous_candidates.object_class with count>=2, is_manipulable=true, and exceeds_weight_limit=false
<ambiguous_stateful_object> ::= an ambiguous_candidates.object_class with count>=2, is_manipulable=true, and is_stateful=true
<ambiguous_openable_object> ::= an ambiguous_stateful_object with "closed" in state
<ambiguous_closable_object> ::= an ambiguous_stateful_object with "open" in state
<ambiguous_turn_on_object> ::= an ambiguous_stateful_object with "off" in state
<ambiguous_turn_off_object> ::= an ambiguous_stateful_object with "on" in state
<ambiguous_fillable_object> ::= an ambiguous_stateful_object with "empty" in state
<ambiguous_emptyable_object> ::= an ambiguous_stateful_object with "full" in state
<shared_color> ::= one value from ambiguous_candidates.ambiguous_attributes.color that is shared by multiple instances of the same object_class
<shared_material> ::= one value from ambiguous_candidates.ambiguous_attributes.material that is shared by multiple instances of the same object_class
<shared_shape> ::= one value from ambiguous_candidates.ambiguous_attributes.shape that is shared by multiple instances of the same object_class
<shared_texture> ::= one value from ambiguous_candidates.ambiguous_attributes.texture that is shared by multiple instances of the same object_class
<shared_pattern> ::= one value from ambiguous_candidates.ambiguous_attributes.pattern that is shared by multiple instances of the same object_class
<shared_condition> ::= one value from ambiguous_candidates.ambiguous_attributes.condition that is shared by multiple instances of the same object_class
<shared_style> ::= one value from ambiguous_candidates.ambiguous_attributes.style that is shared by multiple instances of the same object_class
<shared_size> ::= one value from ambiguous_candidates.size that still matches multiple instances
<on_location> ::= a scene_locations.description whose location_type is surface, shelf, or floor_region and whose size can support the selected object's shared size
<in_location> ::= a scene_locations.description whose location_type is container, drawer, or inside_container and whose size can contain the selected object's shared size
""".strip()


RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        name="pick_up_object",
        pattern="pick up the {object_class}",
        slots=("object_class",),
        notes=(
            "Requires an ambiguous candidate with count>=2, is_manipulable=true, "
            "and exceeds_weight_limit=false."
        ),
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
        name="pick_up_color_object",
        pattern="pick up the {color} {object_class}",
        slots=("color", "object_class"),
        notes=(
            "Requires an ambiguous candidate with is_manipulable=true, "
            "exceeds_weight_limit=false, and at least one ambiguity-preserving color "
            "in ambiguous_attributes.color."
        ),
    ),
    GrammarRule(
        name="pick_up_material_object",
        pattern="pick up the {material} {object_class}",
        slots=("material", "object_class"),
        notes="Requires an ambiguity-preserving material in ambiguous_attributes.material.",
    ),
    GrammarRule(
        name="pick_up_shape_object",
        pattern="pick up the {shape} {object_class}",
        slots=("shape", "object_class"),
        notes="Requires an ambiguity-preserving shape in ambiguous_attributes.shape.",
    ),
    GrammarRule(
        name="pick_up_texture_object",
        pattern="pick up the {texture} {object_class}",
        slots=("texture", "object_class"),
        notes="Requires an ambiguity-preserving texture in ambiguous_attributes.texture.",
    ),
    GrammarRule(
        name="pick_up_pattern_object",
        pattern="pick up the {pattern} {object_class}",
        slots=("pattern", "object_class"),
        notes="Requires an ambiguity-preserving pattern in ambiguous_attributes.pattern.",
    ),
    GrammarRule(
        name="pick_up_condition_object",
        pattern="pick up the {condition} {object_class}",
        slots=("condition", "object_class"),
        notes="Requires an ambiguity-preserving condition in ambiguous_attributes.condition.",
    ),
    GrammarRule(
        name="pick_up_style_object",
        pattern="pick up the {style} {object_class}",
        slots=("style", "object_class"),
        notes="Requires an ambiguity-preserving style in ambiguous_attributes.style.",
    ),
    GrammarRule(
        name="pick_up_size_object",
        pattern="pick up the {size} {object_class}",
        slots=("size", "object_class"),
        notes="Requires at least one shared size value for an ambiguous carryable candidate.",
    ),
    GrammarRule(
        name="put_object_on_location",
        pattern="put the {object_class} on the {location}",
        slots=("object_class", "location"),
        notes=(
            "Requires an ambiguous candidate with is_manipulable=true, "
            "exceeds_weight_limit=false, and at least one visible compatible location. "
            "Compatibility should be checked using the candidate's shared size values."
        ),
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
            "Requires an ambiguous carryable candidate and a compatible container, "
            "drawer, or inside_container location."
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
        notes=(
            "Requires an ambiguous candidate with is_manipulable=true and "
            "exceeds_weight_limit=false."
        ),
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
        notes=(
            "Requires an ambiguous candidate with is_manipulable=true and "
            "exceeds_weight_limit=false."
        ),
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
        notes=(
            'Requires an ambiguous candidate with is_manipulable=true and "closed" '
            'present in its shared state list.'
        ),
    ),
    GrammarRule(
        name="close_object",
        pattern="close the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires an ambiguous candidate with is_manipulable=true and "open" '
            'present in its shared state list.'
        ),
    ),
    GrammarRule(
        name="turn_on_object",
        pattern="turn on the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires an ambiguous candidate with is_manipulable=true, is_stateful=true, '
            'and "off" present in its shared state list.'
        ),
    ),
    GrammarRule(
        name="turn_off_object",
        pattern="turn off the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires an ambiguous candidate with is_manipulable=true, is_stateful=true, '
            'and "on" present in its shared state list.'
        ),
    ),
    GrammarRule(
        name="fill_object",
        pattern="fill the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires an ambiguous candidate with is_manipulable=true, is_stateful=true, '
            'and "empty" present in its shared state list.'
        ),
    ),
    GrammarRule(
        name="empty_object",
        pattern="empty the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires an ambiguous candidate with is_manipulable=true, is_stateful=true, '
            'and "full" present in its shared state list.'
        ),
    ),
)


def get_grammar() -> str:
    return GRAMMAR


def get_rules() -> tuple[GrammarRule, ...]:
    return RULES


ON_LOCATION_TYPES = {"surface", "shelf", "floor_region"}
IN_LOCATION_TYPES = {"container", "drawer", "inside_container"}


def _eligible_manipulable_candidates(
    candidates: list[AmbiguousClass],
    *,
    require_carryable: bool,
) -> list[AmbiguousClass]:
    return [
        candidate
        for candidate in candidates
        if candidate.count >= 2
        and candidate.is_manipulable
        and (not require_carryable or not candidate.exceeds_weight_limit)
    ]


def _eligible_state_candidates(
    scene: Phase2Output,
    *,
    required_state: str,
) -> list[AmbiguousClass]:
    return [
        candidate
        for candidate in _eligible_manipulable_candidates(
            scene.checks.ambiguous_candidates,
            require_carryable=False,
        )
        if candidate.is_stateful and required_state in candidate.state
    ]


def _attribute_values(candidate: AmbiguousClass, attribute_name: str) -> list[str]:
    return list(getattr(candidate.ambiguous_attributes, attribute_name))


def _generate_object_action(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    template: str,
) -> GeneratedInstruction | None:
    candidate = sample_one(
        _eligible_manipulable_candidates(
            scene.checks.ambiguous_candidates,
            require_carryable=True,
        ),
        rng,
    )
    if candidate is None:
        return None
    return GeneratedInstruction(
        instruction=template.format(object_class=format_object_name(candidate.object_class)),
        grammar_rule=grammar_rule,
    )


def _generate_attribute_action(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    attribute_name: str,
) -> GeneratedInstruction | None:
    candidates = [
        candidate
        for candidate in _eligible_manipulable_candidates(
            scene.checks.ambiguous_candidates,
            require_carryable=True,
        )
        if _attribute_values(candidate, attribute_name)
    ]
    candidate = sample_one(candidates, rng)
    if candidate is None:
        return None
    attribute_value = sample_one(_attribute_values(candidate, attribute_name), rng)
    if attribute_value is None:
        return None
    return GeneratedInstruction(
        instruction=(
            f"pick up the {attribute_value} {format_object_name(candidate.object_class)}"
        ),
        grammar_rule=grammar_rule,
    )


def _generate_size_action(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    candidates = [
        candidate
        for candidate in _eligible_manipulable_candidates(
            scene.checks.ambiguous_candidates,
            require_carryable=True,
        )
        if candidate.size
    ]
    candidate = sample_one(candidates, rng)
    if candidate is None:
        return None
    size = sample_one(candidate.size, rng)
    if size is None:
        return None
    return GeneratedInstruction(
        instruction=f"pick up the {size} {format_object_name(candidate.object_class)}",
        grammar_rule="pick_up_size_object",
    )


def _sample_location(
    scene: Phase2Output,
    candidate: AmbiguousClass,
    rng: random.Random,
    *,
    allowed_types: set[str],
) -> SceneLocation | None:
    return sample_location_for_size(
        scene.scene_locations,
        max_shared_size(candidate.size),
        rng,
        allowed_types=allowed_types,
    )


def _generate_location_action(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    template: str,
    allowed_types: set[str],
) -> GeneratedInstruction | None:
    candidates = _eligible_manipulable_candidates(
        scene.checks.ambiguous_candidates,
        require_carryable=True,
    )
    rng.shuffle(candidates)
    for candidate in candidates:
        location = _sample_location(scene, candidate, rng, allowed_types=allowed_types)
        if location is None:
            continue
        return GeneratedInstruction(
            instruction=template.format(
                object_class=format_object_name(candidate.object_class),
                location=location.description,
            ),
            grammar_rule=grammar_rule,
        )
    return None


def _generate_state_action(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    required_state: str,
    template: str,
) -> GeneratedInstruction | None:
    candidate = sample_one(
        _eligible_state_candidates(scene, required_state=required_state),
        rng,
    )
    if candidate is None:
        return None
    return GeneratedInstruction(
        instruction=template.format(object_class=format_object_name(candidate.object_class)),
        grammar_rule=grammar_rule,
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


def _generate_pick_up_color_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_attribute_action(
        scene,
        rng,
        grammar_rule="pick_up_color_object",
        attribute_name="color",
    )


def _generate_pick_up_material_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_attribute_action(
        scene,
        rng,
        grammar_rule="pick_up_material_object",
        attribute_name="material",
    )


def _generate_pick_up_shape_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_attribute_action(
        scene,
        rng,
        grammar_rule="pick_up_shape_object",
        attribute_name="shape",
    )


def _generate_pick_up_texture_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_attribute_action(
        scene,
        rng,
        grammar_rule="pick_up_texture_object",
        attribute_name="texture",
    )


def _generate_pick_up_pattern_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_attribute_action(
        scene,
        rng,
        grammar_rule="pick_up_pattern_object",
        attribute_name="pattern",
    )


def _generate_pick_up_condition_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_attribute_action(
        scene,
        rng,
        grammar_rule="pick_up_condition_object",
        attribute_name="condition",
    )


def _generate_pick_up_style_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_attribute_action(
        scene,
        rng,
        grammar_rule="pick_up_style_object",
        attribute_name="style",
    )


def _generate_pick_up_size_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_size_action(scene, rng)


def _generate_put_object_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="put_object_on_location",
        template="put the {object_class} on the {location}",
        allowed_types=ON_LOCATION_TYPES,
    )


def _generate_place_object_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="place_object_on_location",
        template="place the {object_class} on the {location}",
        allowed_types=ON_LOCATION_TYPES,
    )


def _generate_set_object_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="set_object_on_location",
        template="set the {object_class} on the {location}",
        allowed_types=ON_LOCATION_TYPES,
    )


def _generate_put_object_in_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="put_object_in_location",
        template="put the {object_class} in the {location}",
        allowed_types=IN_LOCATION_TYPES,
    )


def _generate_place_object_in_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_action(
        scene,
        rng,
        grammar_rule="place_object_in_location",
        template="place the {object_class} in the {location}",
        allowed_types=IN_LOCATION_TYPES,
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
    "pick_up_object": _generate_pick_up_object,
    "grab_object": _generate_grab_object,
    "take_object": _generate_take_object,
    "move_object": _generate_move_object,
    "pick_up_color_object": _generate_pick_up_color_object,
    "pick_up_material_object": _generate_pick_up_material_object,
    "pick_up_shape_object": _generate_pick_up_shape_object,
    "pick_up_texture_object": _generate_pick_up_texture_object,
    "pick_up_pattern_object": _generate_pick_up_pattern_object,
    "pick_up_condition_object": _generate_pick_up_condition_object,
    "pick_up_style_object": _generate_pick_up_style_object,
    "pick_up_size_object": _generate_pick_up_size_object,
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
