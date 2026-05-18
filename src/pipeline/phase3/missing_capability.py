from __future__ import annotations

import random
from dataclasses import asdict

from pipeline.phase2.models import Phase2Output, SceneObject
from pipeline.phase3.helper import (
    GeneratedInstruction,
    GrammarRule,
    format_object_name,
    sample_one,
)


SUPPORTED_ROBOT_MODALITIES = {"vision", "manipulation"}
MISSING_MODALITIES = {
    "olfaction",
    "audition",
    "proprioception",
    "thermal_sensing",
}


GRAMMAR = """
<missing_capability> ::= <olfactory_query>
                        | <auditory_query>
                        | <proprioceptive_query>
                        | <proprioceptive_conditional_action>
                        | <thermal_query>
                        | <thermal_conditional_action>

<olfactory_query> ::= <olfactory_smells_bad>
                    | <olfactory_has_odor>
                    | <olfactory_smells_clean>

<auditory_query> ::= <auditory_making_noise>
                   | <auditory_beeping>
                   | <auditory_ringing>
                   | <auditory_buzzing>
                   | <auditory_humming>

<proprioceptive_query> ::= <proprioceptive_feels_heavy>
                         | <proprioceptive_hard_to_move>
                         | <proprioceptive_stuck>
                         | <proprioceptive_feels_soft>
                         | <proprioceptive_feels_firm>
                         | <proprioceptive_resists_movement>

<proprioceptive_conditional_action> ::= <move_if_feels_light>
                                      | <pick_up_if_feels_light>
                                      | <bring_if_feels_light>
                                      | <open_if_easy_to_do>
                                      | <close_if_easy_to_do>
                                      | <open_if_moves_smoothly>
                                      | <close_if_moves_smoothly>

<thermal_query> ::= <thermal_is_hot>
                  | <thermal_is_warm>
                  | <thermal_is_cold>
                  | <thermal_cooled_down>
                  | <thermal_still_warm>
                  | <thermal_too_hot_to_touch>

<thermal_conditional_action> ::= <move_if_cool>
                               | <bring_if_cool>
                               | <hand_if_cool>
                               | <pick_up_if_cool>
                               | <move_if_not_hot>
                               | <bring_if_not_hot>
                               | <hand_if_not_hot>
                               | <pick_up_if_not_hot>
                               | <move_if_not_warm>
                               | <bring_if_not_warm>
                               | <hand_if_not_warm>
                               | <pick_up_if_not_warm>
                               | <move_if_cooled_down>
                               | <bring_if_cooled_down>
                               | <hand_if_cooled_down>
                               | <pick_up_if_cooled_down>
                               | <open_if_not_hot>
                               | <close_if_not_hot>

<olfactory_smells_bad> ::= <query_prefix> " the " <olfactory_object> " smells bad"
<olfactory_has_odor> ::= <query_prefix> " the " <olfactory_object> " has an odor"
<olfactory_smells_clean> ::= <query_prefix> " the " <olfactory_object> " smells clean"

<auditory_making_noise> ::= <query_prefix> " the " <auditory_object> " is making noise"
<auditory_beeping> ::= <query_prefix> " the " <auditory_object> " is beeping"
<auditory_ringing> ::= <query_prefix> " the " <auditory_object> " is ringing"
<auditory_buzzing> ::= <query_prefix> " the " <auditory_object> " is buzzing"
<auditory_humming> ::= <query_prefix> " the " <auditory_object> " is humming"

<proprioceptive_feels_heavy> ::= <query_prefix> " the " <proprioceptive_manipulable_object> " feels heavy"
<proprioceptive_hard_to_move> ::= <query_prefix> " the " <proprioceptive_manipulable_object> " is hard to move"
<proprioceptive_stuck> ::= <query_prefix> " the " <proprioceptive_manipulable_object> " is stuck"
<proprioceptive_feels_soft> ::= <query_prefix> " the " <proprioceptive_manipulable_object> " feels soft"
<proprioceptive_feels_firm> ::= <query_prefix> " the " <proprioceptive_manipulable_object> " feels firm"
<proprioceptive_resists_movement> ::= <query_prefix> " the " <proprioceptive_manipulable_object> " resists movement"

<move_if_feels_light> ::= "move the " <proprioceptive_carryable_object> " only if it feels light"
<pick_up_if_feels_light> ::= "pick up the " <proprioceptive_carryable_object> " only if it feels light"
<bring_if_feels_light> ::= "bring me the " <proprioceptive_carryable_object> " only if it feels light"
<open_if_easy_to_do> ::= "open the " <proprioceptive_closed_object> " only if it is easy to do"
<close_if_easy_to_do> ::= "close the " <proprioceptive_open_object> " only if it is easy to do"
<open_if_moves_smoothly> ::= "open the " <proprioceptive_closed_object> " only if it moves smoothly"
<close_if_moves_smoothly> ::= "close the " <proprioceptive_open_object> " only if it moves smoothly"

<thermal_is_hot> ::= <query_prefix> " the " <thermal_object> " is hot"
<thermal_is_warm> ::= <query_prefix> " the " <thermal_object> " is warm"
<thermal_is_cold> ::= <query_prefix> " the " <thermal_object> " is cold"
<thermal_cooled_down> ::= <query_prefix> " the " <thermal_object> " has cooled down"
<thermal_still_warm> ::= <query_prefix> " the " <thermal_object> " is still warm"
<thermal_too_hot_to_touch> ::= "check if the " <thermal_object> " is too hot to touch"

<move_if_cool> ::= "move the " <thermal_carryable_object> " only if it is cool"
<bring_if_cool> ::= "bring me the " <thermal_carryable_object> " only if it is cool"
<hand_if_cool> ::= "hand me the " <thermal_carryable_object> " only if it is cool"
<pick_up_if_cool> ::= "pick up the " <thermal_carryable_object> " only if it is cool"
<move_if_not_hot> ::= "move the " <thermal_carryable_object> " only if it is not hot"
<bring_if_not_hot> ::= "bring me the " <thermal_carryable_object> " only if it is not hot"
<hand_if_not_hot> ::= "hand me the " <thermal_carryable_object> " only if it is not hot"
<pick_up_if_not_hot> ::= "pick up the " <thermal_carryable_object> " only if it is not hot"
<move_if_not_warm> ::= "move the " <thermal_carryable_object> " only if it is not warm"
<bring_if_not_warm> ::= "bring me the " <thermal_carryable_object> " only if it is not warm"
<hand_if_not_warm> ::= "hand me the " <thermal_carryable_object> " only if it is not warm"
<pick_up_if_not_warm> ::= "pick up the " <thermal_carryable_object> " only if it is not warm"
<move_if_cooled_down> ::= "move the " <thermal_carryable_object> " only if it has cooled down"
<bring_if_cooled_down> ::= "bring me the " <thermal_carryable_object> " only if it has cooled down"
<hand_if_cooled_down> ::= "hand me the " <thermal_carryable_object> " only if it has cooled down"
<pick_up_if_cooled_down> ::= "pick up the " <thermal_carryable_object> " only if it has cooled down"
<open_if_not_hot> ::= "open the " <thermal_closed_object> " only if it is not hot"
<close_if_not_hot> ::= "close the " <thermal_open_object> " only if it is not hot"

<query_prefix> ::= "tell me if" | "tell me whether" | "check if" | "check whether"

<olfactory_object> ::= a non-ambiguous scene_object whose missing_capability_candidate.required_modality is "olfaction"
<auditory_object> ::= a non-ambiguous scene_object whose missing_capability_candidate.required_modality is "audition"
<thermal_object> ::= a non-ambiguous scene_object whose missing_capability_candidate.required_modality is "thermal_sensing"

<proprioceptive_manipulable_object> ::= a non-ambiguous scene_object whose missing_capability_candidate.required_modality is "proprioception" and is_manipulable=true
<proprioceptive_carryable_object> ::= a proprioceptive_manipulable_object with exceeds_weight_limit=false
<proprioceptive_closed_object> ::= a proprioceptive_carryable_object with is_stateful=true and state="closed"
<proprioceptive_open_object> ::= a proprioceptive_carryable_object with is_stateful=true and state="open"

<thermal_carryable_object> ::= a thermal_object with is_manipulable=true and exceeds_weight_limit=false
<thermal_closed_object> ::= a thermal_carryable_object with is_stateful=true and state="closed"
<thermal_open_object> ::= a thermal_carryable_object with is_stateful=true and state="open"

Non-ambiguity constraint:
- object_class must not appear in checks.ambiguous_candidates
- do not use absent objects
- do not use selection forms such as "find the object that ..."
- do not use comparative forms such as hottest, loudest, heaviest, or strongest smell
""".strip()


def _rule(
    name: str,
    pattern: str,
    *,
    modality: str,
    slots: tuple[str, ...] = ("object_class",),
    extra_notes: str = "",
) -> GrammarRule:
    notes = (
        f'Requires a non-ambiguous scene object with required_modality="{modality}".'
    )
    if extra_notes:
        notes = f"{notes} {extra_notes}"
    return GrammarRule(name=name, pattern=pattern, slots=slots, notes=notes)


RULES: tuple[GrammarRule, ...] = (
    _rule(
        "olfactory_smells_bad",
        "{query_prefix} the {object_class} smells bad",
        modality="olfaction",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "olfactory_has_odor",
        "{query_prefix} the {object_class} has an odor",
        modality="olfaction",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "olfactory_smells_clean",
        "{query_prefix} the {object_class} smells clean",
        modality="olfaction",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "auditory_making_noise",
        "{query_prefix} the {object_class} is making noise",
        modality="audition",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "auditory_beeping",
        "{query_prefix} the {object_class} is beeping",
        modality="audition",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "auditory_ringing",
        "{query_prefix} the {object_class} is ringing",
        modality="audition",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "auditory_buzzing",
        "{query_prefix} the {object_class} is buzzing",
        modality="audition",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "auditory_humming",
        "{query_prefix} the {object_class} is humming",
        modality="audition",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "proprioceptive_feels_heavy",
        "{query_prefix} the {object_class} feels heavy",
        modality="proprioception",
        slots=("query_prefix", "object_class"),
        extra_notes="Requires is_manipulable=true.",
    ),
    _rule(
        "proprioceptive_hard_to_move",
        "{query_prefix} the {object_class} is hard to move",
        modality="proprioception",
        slots=("query_prefix", "object_class"),
        extra_notes="Requires is_manipulable=true.",
    ),
    _rule(
        "proprioceptive_stuck",
        "{query_prefix} the {object_class} is stuck",
        modality="proprioception",
        slots=("query_prefix", "object_class"),
        extra_notes="Requires is_manipulable=true.",
    ),
    _rule(
        "proprioceptive_feels_soft",
        "{query_prefix} the {object_class} feels soft",
        modality="proprioception",
        slots=("query_prefix", "object_class"),
        extra_notes="Requires is_manipulable=true.",
    ),
    _rule(
        "proprioceptive_feels_firm",
        "{query_prefix} the {object_class} feels firm",
        modality="proprioception",
        slots=("query_prefix", "object_class"),
        extra_notes="Requires is_manipulable=true.",
    ),
    _rule(
        "proprioceptive_resists_movement",
        "{query_prefix} the {object_class} resists movement",
        modality="proprioception",
        slots=("query_prefix", "object_class"),
        extra_notes="Requires is_manipulable=true.",
    ),
    _rule(
        "move_if_feels_light",
        "move the {object_class} only if it feels light",
        modality="proprioception",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "pick_up_if_feels_light",
        "pick up the {object_class} only if it feels light",
        modality="proprioception",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "bring_if_feels_light",
        "bring me the {object_class} only if it feels light",
        modality="proprioception",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "open_if_easy_to_do",
        "open the {object_class} only if it is easy to do",
        modality="proprioception",
        extra_notes=(
            'Requires is_manipulable=true, exceeds_weight_limit=false, '
            'is_stateful=true, and state="closed".'
        ),
    ),
    _rule(
        "close_if_easy_to_do",
        "close the {object_class} only if it is easy to do",
        modality="proprioception",
        extra_notes=(
            'Requires is_manipulable=true, exceeds_weight_limit=false, '
            'is_stateful=true, and state="open".'
        ),
    ),
    _rule(
        "open_if_moves_smoothly",
        "open the {object_class} only if it moves smoothly",
        modality="proprioception",
        extra_notes=(
            'Requires is_manipulable=true, exceeds_weight_limit=false, '
            'is_stateful=true, and state="closed".'
        ),
    ),
    _rule(
        "close_if_moves_smoothly",
        "close the {object_class} only if it moves smoothly",
        modality="proprioception",
        extra_notes=(
            'Requires is_manipulable=true, exceeds_weight_limit=false, '
            'is_stateful=true, and state="open".'
        ),
    ),
    _rule(
        "thermal_is_hot",
        "{query_prefix} the {object_class} is hot",
        modality="thermal_sensing",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "thermal_is_warm",
        "{query_prefix} the {object_class} is warm",
        modality="thermal_sensing",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "thermal_is_cold",
        "{query_prefix} the {object_class} is cold",
        modality="thermal_sensing",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "thermal_cooled_down",
        "{query_prefix} the {object_class} has cooled down",
        modality="thermal_sensing",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "thermal_still_warm",
        "{query_prefix} the {object_class} is still warm",
        modality="thermal_sensing",
        slots=("query_prefix", "object_class"),
    ),
    _rule(
        "thermal_too_hot_to_touch",
        "check if the {object_class} is too hot to touch",
        modality="thermal_sensing",
    ),
    _rule(
        "move_if_cool",
        "move the {object_class} only if it is cool",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "bring_if_cool",
        "bring me the {object_class} only if it is cool",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "hand_if_cool",
        "hand me the {object_class} only if it is cool",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "pick_up_if_cool",
        "pick up the {object_class} only if it is cool",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "move_if_not_hot",
        "move the {object_class} only if it is not hot",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "bring_if_not_hot",
        "bring me the {object_class} only if it is not hot",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "hand_if_not_hot",
        "hand me the {object_class} only if it is not hot",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "pick_up_if_not_hot",
        "pick up the {object_class} only if it is not hot",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "move_if_not_warm",
        "move the {object_class} only if it is not warm",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "bring_if_not_warm",
        "bring me the {object_class} only if it is not warm",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "hand_if_not_warm",
        "hand me the {object_class} only if it is not warm",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "pick_up_if_not_warm",
        "pick up the {object_class} only if it is not warm",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "move_if_cooled_down",
        "move the {object_class} only if it has cooled down",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "bring_if_cooled_down",
        "bring me the {object_class} only if it has cooled down",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "hand_if_cooled_down",
        "hand me the {object_class} only if it has cooled down",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "pick_up_if_cooled_down",
        "pick up the {object_class} only if it has cooled down",
        modality="thermal_sensing",
        extra_notes="Requires is_manipulable=true and exceeds_weight_limit=false.",
    ),
    _rule(
        "open_if_not_hot",
        "open the {object_class} only if it is not hot",
        modality="thermal_sensing",
        extra_notes=(
            'Requires is_manipulable=true, exceeds_weight_limit=false, '
            'is_stateful=true, and state="closed".'
        ),
    ),
    _rule(
        "close_if_not_hot",
        "close the {object_class} only if it is not hot",
        modality="thermal_sensing",
        extra_notes=(
            'Requires is_manipulable=true, exceeds_weight_limit=false, '
            'is_stateful=true, and state="open".'
        ),
    ),
)


def get_grammar() -> str:
    return GRAMMAR


def get_rules() -> tuple[GrammarRule, ...]:
    return RULES


QUERY_PREFIXES = (
    "tell me if",
    "tell me whether",
    "check if",
    "check whether",
)


QUERY_RULE_TEMPLATES: dict[str, tuple[str, str, bool]] = {
    "olfactory_smells_bad": ("olfaction", "{query_prefix} the {object_class} smells bad", False),
    "olfactory_has_odor": ("olfaction", "{query_prefix} the {object_class} has an odor", False),
    "olfactory_smells_clean": ("olfaction", "{query_prefix} the {object_class} smells clean", False),
    "auditory_making_noise": ("audition", "{query_prefix} the {object_class} is making noise", False),
    "auditory_beeping": ("audition", "{query_prefix} the {object_class} is beeping", False),
    "auditory_ringing": ("audition", "{query_prefix} the {object_class} is ringing", False),
    "auditory_buzzing": ("audition", "{query_prefix} the {object_class} is buzzing", False),
    "auditory_humming": ("audition", "{query_prefix} the {object_class} is humming", False),
    "proprioceptive_feels_heavy": ("proprioception", "{query_prefix} the {object_class} feels heavy", True),
    "proprioceptive_hard_to_move": ("proprioception", "{query_prefix} the {object_class} is hard to move", True),
    "proprioceptive_stuck": ("proprioception", "{query_prefix} the {object_class} is stuck", True),
    "proprioceptive_feels_soft": ("proprioception", "{query_prefix} the {object_class} feels soft", True),
    "proprioceptive_feels_firm": ("proprioception", "{query_prefix} the {object_class} feels firm", True),
    "proprioceptive_resists_movement": ("proprioception", "{query_prefix} the {object_class} resists movement", True),
    "thermal_is_hot": ("thermal_sensing", "{query_prefix} the {object_class} is hot", False),
    "thermal_is_warm": ("thermal_sensing", "{query_prefix} the {object_class} is warm", False),
    "thermal_is_cold": ("thermal_sensing", "{query_prefix} the {object_class} is cold", False),
    "thermal_cooled_down": ("thermal_sensing", "{query_prefix} the {object_class} has cooled down", False),
    "thermal_still_warm": ("thermal_sensing", "{query_prefix} the {object_class} is still warm", False),
    "thermal_too_hot_to_touch": ("thermal_sensing", "check if the {object_class} is too hot to touch", False),
}


CONDITIONAL_RULE_TEMPLATES: dict[str, tuple[str, str, str | None]] = {
    "move_if_feels_light": ("proprioception", "move the {object_class} only if it feels light", None),
    "pick_up_if_feels_light": ("proprioception", "pick up the {object_class} only if it feels light", None),
    "bring_if_feels_light": ("proprioception", "bring me the {object_class} only if it feels light", None),
    "open_if_easy_to_do": ("proprioception", "open the {object_class} only if it is easy to do", "closed"),
    "close_if_easy_to_do": ("proprioception", "close the {object_class} only if it is easy to do", "open"),
    "open_if_moves_smoothly": ("proprioception", "open the {object_class} only if it moves smoothly", "closed"),
    "close_if_moves_smoothly": ("proprioception", "close the {object_class} only if it moves smoothly", "open"),
    "move_if_cool": ("thermal_sensing", "move the {object_class} only if it is cool", None),
    "bring_if_cool": ("thermal_sensing", "bring me the {object_class} only if it is cool", None),
    "hand_if_cool": ("thermal_sensing", "hand me the {object_class} only if it is cool", None),
    "pick_up_if_cool": ("thermal_sensing", "pick up the {object_class} only if it is cool", None),
    "move_if_not_hot": ("thermal_sensing", "move the {object_class} only if it is not hot", None),
    "bring_if_not_hot": ("thermal_sensing", "bring me the {object_class} only if it is not hot", None),
    "hand_if_not_hot": ("thermal_sensing", "hand me the {object_class} only if it is not hot", None),
    "pick_up_if_not_hot": ("thermal_sensing", "pick up the {object_class} only if it is not hot", None),
    "move_if_not_warm": ("thermal_sensing", "move the {object_class} only if it is not warm", None),
    "bring_if_not_warm": ("thermal_sensing", "bring me the {object_class} only if it is not warm", None),
    "hand_if_not_warm": ("thermal_sensing", "hand me the {object_class} only if it is not warm", None),
    "pick_up_if_not_warm": ("thermal_sensing", "pick up the {object_class} only if it is not warm", None),
    "move_if_cooled_down": ("thermal_sensing", "move the {object_class} only if it has cooled down", None),
    "bring_if_cooled_down": ("thermal_sensing", "bring me the {object_class} only if it has cooled down", None),
    "hand_if_cooled_down": ("thermal_sensing", "hand me the {object_class} only if it has cooled down", None),
    "pick_up_if_cooled_down": ("thermal_sensing", "pick up the {object_class} only if it has cooled down", None),
    "open_if_not_hot": ("thermal_sensing", "open the {object_class} only if it is not hot", "closed"),
    "close_if_not_hot": ("thermal_sensing", "close the {object_class} only if it is not hot", "open"),
}


def _object_by_id(scene: Phase2Output) -> dict[str, SceneObject]:
    return {obj.id: obj for obj in scene.scene_objects}


def _ambiguous_classes(scene: Phase2Output) -> set[str]:
    return {candidate.object_class for candidate in scene.checks.ambiguous_candidates}


def _objects_for_modality(
    scene: Phase2Output,
    modality: str,
    *,
    require_manipulable: bool,
    require_carryable: bool,
    state: str | None = None,
) -> list[SceneObject]:
    objects_by_id = _object_by_id(scene)
    ambiguous_classes = _ambiguous_classes(scene)
    results: list[SceneObject] = []
    seen_object_ids: set[str] = set()

    for candidate in scene.checks.missing_capability_candidates:
        if candidate.required_modality != modality:
            continue
        if candidate.object_id in seen_object_ids:
            continue
        obj = objects_by_id.get(candidate.object_id)
        if obj is None:
            continue
        if obj.object_class in ambiguous_classes:
            continue
        if require_manipulable and not obj.is_manipulable:
            continue
        if require_carryable and obj.exceeds_weight_limit:
            continue
        if state is not None and (not obj.is_stateful or obj.state != state):
            continue
        seen_object_ids.add(candidate.object_id)
        results.append(obj)
    return results


def _generate_query_rule(
    scene: Phase2Output,
    rng: random.Random,
    *,
    rule_name: str,
) -> GeneratedInstruction | None:
    modality, template, require_manipulable = QUERY_RULE_TEMPLATES[rule_name]
    obj = sample_one(
        _objects_for_modality(
            scene,
            modality,
            require_manipulable=require_manipulable,
            require_carryable=False,
        ),
        rng,
    )
    if obj is None:
        return None
    query_prefix = sample_one(list(QUERY_PREFIXES), rng)
    instruction = template.format(
        query_prefix=query_prefix,
        object_class=format_object_name(obj.object_class),
    )
    return GeneratedInstruction(instruction=instruction, grammar_rule=rule_name)


def _generate_conditional_rule(
    scene: Phase2Output,
    rng: random.Random,
    *,
    rule_name: str,
) -> GeneratedInstruction | None:
    modality, template, state = CONDITIONAL_RULE_TEMPLATES[rule_name]
    obj = sample_one(
        _objects_for_modality(
            scene,
            modality,
            require_manipulable=True,
            require_carryable=True,
            state=state,
        ),
        rng,
    )
    if obj is None:
        return None
    instruction = template.format(object_class=format_object_name(obj.object_class))
    return GeneratedInstruction(instruction=instruction, grammar_rule=rule_name)


def generate_instructions(scene: Phase2Output, seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)
    generated: list[GeneratedInstruction] = []
    seen_instructions: set[str] = set()

    for rule in RULES:
        if rule.name in QUERY_RULE_TEMPLATES:
            result = _generate_query_rule(scene, rng, rule_name=rule.name)
        else:
            result = _generate_conditional_rule(scene, rng, rule_name=rule.name)
        if result is None:
            continue
        if result.instruction in seen_instructions:
            continue
        seen_instructions.add(result.instruction)
        generated.append(result)

    return [asdict(item) for item in generated]
