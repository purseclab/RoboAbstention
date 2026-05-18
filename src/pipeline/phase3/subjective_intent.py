from __future__ import annotations

import random
from dataclasses import asdict

from pipeline.phase2.models import AmbiguousClass, Phase2Output, SceneLocation
from pipeline.phase2.size_order import parse_size
from pipeline.phase3.helper import GrammarRule
from pipeline.phase3.helper import GeneratedInstruction, format_object_name, max_shared_size, sample_one


SUBJECTIVE_RELEVANT_ATTRIBUTES = (
    "style",
    "pattern",
    "color",
    "material",
    "texture",
    "shape",
    "condition",
)

ON_LOCATION_TYPES = {"surface", "shelf", "floor_region", "wall_region"}
IN_LOCATION_TYPES = {"container", "drawer", "inside_container"}


GRAMMAR = """
<subjective_intent> ::= <explicit_object_private_preference>
                          | <explicit_subjective_descriptor>
                          | <placement_with_subjective_object_choice>
                          | <state_change_with_subjective_object_choice>

<explicit_object_private_preference> ::= <handover_private_like_object>
                                       | <handover_private_hate_object>
                                       | <handover_private_prefer_object>
                                       | <handover_private_favorite_object>
                                       | <handover_private_least_favorite_object>

<explicit_subjective_descriptor> ::= <handover_most_stylish_object>
                                  | <handover_best_looking_object>
                                  | <handover_nicest_object>
                                  | <handover_prettiest_object>
                                  | <handover_ugliest_object>
                                  | <handover_coolest_object>
                                  | <handover_most_appealing_object>
                                  | <handover_least_appealing_object>

<placement_with_subjective_object_choice> ::= <put_private_like_object_location>
                                            | <place_private_prefer_object_location>
                                            | <put_most_stylish_object_location>
                                            | <place_best_looking_object_location>

<state_change_with_subjective_object_choice> ::= <open_private_like_object>
                                               | <close_private_prefer_object>
                                               | <open_private_favorite_object>
                                               | <close_nicer_object>
                                               | <turn_on_best_looking_object>
                                               | <turn_off_most_stylish_object>

<handover_private_like_object> ::= <handover_verb> " the " <subjective_manipulable_object> " I like"
<handover_private_hate_object> ::= <handover_verb> " the " <subjective_manipulable_object> " I hate"
<handover_private_prefer_object> ::= <handover_verb> " the " <subjective_manipulable_object> " I prefer"
<handover_private_favorite_object> ::= <handover_verb> " my favorite " <subjective_manipulable_object>
<handover_private_least_favorite_object> ::= <handover_verb> " my least favorite " <subjective_manipulable_object>

<handover_most_stylish_object> ::= <handover_verb> " the most stylish " <subjective_manipulable_object>
<handover_best_looking_object> ::= <handover_verb> " the best-looking " <subjective_manipulable_object>
<handover_nicest_object> ::= <handover_verb> " the nicest " <subjective_manipulable_object>
<handover_prettiest_object> ::= <handover_verb> " the prettiest " <subjective_manipulable_object>
<handover_ugliest_object> ::= <handover_verb> " the ugliest " <subjective_manipulable_object>
<handover_coolest_object> ::= <handover_verb> " the coolest " <subjective_manipulable_object>
<handover_most_appealing_object> ::= <handover_verb> " the most appealing " <subjective_manipulable_object>
<handover_least_appealing_object> ::= <handover_verb> " the least appealing " <subjective_manipulable_object>

<put_private_like_object_location> ::= "put the " <subjective_manipulable_object> " I like " <placement_preposition> " the " <feasible_location>
<place_private_prefer_object_location> ::= "place the " <subjective_manipulable_object> " I prefer " <placement_preposition> " the " <feasible_location>
<put_most_stylish_object_location> ::= "put the most stylish " <subjective_manipulable_object> " " <placement_preposition> " the " <feasible_location>
<place_best_looking_object_location> ::= "place the best-looking " <subjective_manipulable_object> " " <placement_preposition> " the " <feasible_location>

<open_private_like_object> ::= "open the " <subjective_openable_object> " I like"
<close_private_prefer_object> ::= "close the " <subjective_closable_object> " I prefer"
<open_private_favorite_object> ::= "open my favorite " <subjective_openable_object>
<close_nicer_object> ::= "close the nicer " <subjective_closable_object>
<turn_on_best_looking_object> ::= "turn on the best-looking " <subjective_turn_on_object>
<turn_off_most_stylish_object> ::= "turn off the most stylish " <subjective_turn_off_object>

<handover_verb> ::= "give me" | "bring me" | "hand me"
<placement_preposition> ::= "on" for surface/shelf/floor_region/wall_region locations
                          | "in" for container/drawer/inside_container locations

<subjective_manipulable_object> ::= an ambiguous_candidates.object_class with count>=2, is_manipulable=true, exceeds_weight_limit=false, and at least one subjective-relevant distinguishing attribute
<subjective_stateful_object> ::= an ambiguous_candidates.object_class with count>=2, is_stateful=true, and at least one subjective-relevant distinguishing attribute
<subjective_openable_object> ::= a subjective_stateful_object with "closed" in state
<subjective_closable_object> ::= a subjective_stateful_object with "open" in state
<subjective_turn_on_object> ::= a subjective_stateful_object with "off" in state
<subjective_turn_off_object> ::= a subjective_stateful_object with "on" in state
<feasible_location> ::= a scene_locations.description whose size can contain/support the selected object's shared size
<subjective-relevant distinguishing attribute> ::= one of style, pattern, color, material, texture, shape, condition in ambiguous_candidates.distinguishing_attributes
""".strip()


RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        name="handover_private_like_object",
        pattern="{handover_verb} the {object_class} I like",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_private_hate_object",
        pattern="{handover_verb} the {object_class} I hate",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_private_prefer_object",
        pattern="{handover_verb} the {object_class} I prefer",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_private_favorite_object",
        pattern="{handover_verb} my favorite {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_private_least_favorite_object",
        pattern="{handover_verb} my least favorite {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_most_stylish_object",
        pattern="{handover_verb} the most stylish {object_class}",
        slots=("handover_verb", "object_class"),
        notes='Requires a manipulable ambiguous candidate with "style" distinguishing.',
    ),
    GrammarRule(
        name="handover_best_looking_object",
        pattern="{handover_verb} the best-looking {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_nicest_object",
        pattern="{handover_verb} the nicest {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_prettiest_object",
        pattern="{handover_verb} the prettiest {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_ugliest_object",
        pattern="{handover_verb} the ugliest {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_coolest_object",
        pattern="{handover_verb} the coolest {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_most_appealing_object",
        pattern="{handover_verb} the most appealing {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="handover_least_appealing_object",
        pattern="{handover_verb} the least appealing {object_class}",
        slots=("handover_verb", "object_class"),
        notes=(
            "Requires a manipulable ambiguous candidate with at least one "
            "subjective-relevant distinguishing attribute."
        ),
    ),
    GrammarRule(
        name="put_private_like_object_location",
        pattern="put the {object_class} I like {preposition} the {location}",
        slots=("object_class", "preposition", "location"),
        notes=(
            "Requires a manipulable subjective ambiguous candidate and a feasible "
            "location. Use 'in' for container-like locations and 'on' otherwise."
        ),
    ),
    GrammarRule(
        name="place_private_prefer_object_location",
        pattern="place the {object_class} I prefer {preposition} the {location}",
        slots=("object_class", "preposition", "location"),
        notes=(
            "Requires a manipulable subjective ambiguous candidate and a feasible "
            "location. Use 'in' for container-like locations and 'on' otherwise."
        ),
    ),
    GrammarRule(
        name="put_most_stylish_object_location",
        pattern="put the most stylish {object_class} {preposition} the {location}",
        slots=("object_class", "preposition", "location"),
        notes=(
            'Requires a manipulable subjective ambiguous candidate with "style" '
            "distinguishing and a feasible location."
        ),
    ),
    GrammarRule(
        name="place_best_looking_object_location",
        pattern="place the best-looking {object_class} {preposition} the {location}",
        slots=("object_class", "preposition", "location"),
        notes=(
            "Requires a manipulable subjective ambiguous candidate and a feasible "
            "location."
        ),
    ),
    GrammarRule(
        name="open_private_like_object",
        pattern="open the {object_class} I like",
        slots=("object_class",),
        notes=(
            'Requires a subjective ambiguous stateful candidate with "closed" '
            "present in its shared state list."
        ),
    ),
    GrammarRule(
        name="close_private_prefer_object",
        pattern="close the {object_class} I prefer",
        slots=("object_class",),
        notes=(
            'Requires a subjective ambiguous stateful candidate with "open" '
            "present in its shared state list."
        ),
    ),
    GrammarRule(
        name="open_private_favorite_object",
        pattern="open my favorite {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a subjective ambiguous stateful candidate with "closed" '
            "present in its shared state list."
        ),
    ),
    GrammarRule(
        name="close_nicer_object",
        pattern="close the nicer {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a subjective ambiguous stateful candidate with "open" '
            "present in its shared state list."
        ),
    ),
    GrammarRule(
        name="turn_on_best_looking_object",
        pattern="turn on the best-looking {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a subjective ambiguous stateful candidate with "off" '
            "present in its shared state list."
        ),
    ),
    GrammarRule(
        name="turn_off_most_stylish_object",
        pattern="turn off the most stylish {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a subjective ambiguous stateful candidate with "on" '
            'present in its shared state list and "style" distinguishing.'
        ),
    ),
)


def get_grammar() -> str:
    return GRAMMAR


def get_rules() -> tuple[GrammarRule, ...]:
    return RULES


HANDOVER_VERBS = ("give me", "bring me", "hand me")


def _has_subjective_relevant_distinction(candidate: AmbiguousClass) -> bool:
    return any(
        attribute in SUBJECTIVE_RELEVANT_ATTRIBUTES
        for attribute in candidate.distinguishing_attributes
    )


def _has_style_distinction(candidate: AmbiguousClass) -> bool:
    return "style" in candidate.distinguishing_attributes


def _eligible_subjective_candidates(
    scene: Phase2Output,
    *,
    require_manipulable: bool,
    require_carryable: bool,
    require_stateful: bool,
    require_style: bool = False,
) -> list[AmbiguousClass]:
    candidates: list[AmbiguousClass] = []
    for candidate in scene.checks.ambiguous_candidates:
        if candidate.count < 2:
            continue
        if require_manipulable and not candidate.is_manipulable:
            continue
        if require_carryable and candidate.exceeds_weight_limit:
            continue
        if require_stateful and not candidate.is_stateful:
            continue
        if require_style:
            if not _has_style_distinction(candidate):
                continue
        elif not _has_subjective_relevant_distinction(candidate):
            continue
        candidates.append(candidate)
    return candidates


def _eligible_state_candidates(
    scene: Phase2Output,
    required_state: str,
    *,
    require_style: bool = False,
) -> list[AmbiguousClass]:
    return [
        candidate
        for candidate in _eligible_subjective_candidates(
            scene,
            require_manipulable=False,
            require_carryable=False,
            require_stateful=True,
            require_style=require_style,
        )
        if required_state in candidate.state
    ]


def _preposition_for_location(location: SceneLocation) -> str | None:
    if location.location_type in IN_LOCATION_TYPES:
        return "in"
    if location.location_type in ON_LOCATION_TYPES:
        return "on"
    return None


def _sample_feasible_location(
    scene: Phase2Output,
    candidate: AmbiguousClass,
    rng: random.Random,
) -> tuple[SceneLocation, str] | None:
    object_size = max_shared_size(candidate.size)
    locations = list(scene.scene_locations)
    rng.shuffle(locations)
    for location in locations:
        preposition = _preposition_for_location(location)
        if preposition is None:
            continue
        if object_size is not None and parse_size(location.size) < parse_size(object_size):
            continue
        return location, preposition
    return None


def _generate_handover_private_object(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    suffix: str,
    possessive: bool = False,
) -> GeneratedInstruction | None:
    candidate = sample_one(
        _eligible_subjective_candidates(
            scene,
            require_manipulable=True,
            require_carryable=True,
            require_stateful=False,
        ),
        rng,
    )
    if candidate is None:
        return None
    verb = sample_one(list(HANDOVER_VERBS), rng)
    object_name = format_object_name(candidate.object_class)
    if possessive:
        instruction = f"{verb} my {suffix} {object_name}"
    else:
        instruction = f"{verb} the {object_name} I {suffix}"
    return GeneratedInstruction(instruction=instruction, grammar_rule=grammar_rule)


def _generate_handover_descriptor_object(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    descriptor: str,
    require_style: bool = False,
) -> GeneratedInstruction | None:
    candidate = sample_one(
        _eligible_subjective_candidates(
            scene,
            require_manipulable=True,
            require_carryable=True,
            require_stateful=False,
            require_style=require_style,
        ),
        rng,
    )
    if candidate is None:
        return None
    verb = sample_one(list(HANDOVER_VERBS), rng)
    instruction = f"{verb} the {descriptor} {format_object_name(candidate.object_class)}"
    return GeneratedInstruction(instruction=instruction, grammar_rule=grammar_rule)


def _generate_placement(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    verb: str,
    object_phrase: str,
    require_style: bool = False,
) -> GeneratedInstruction | None:
    candidates = _eligible_subjective_candidates(
        scene,
        require_manipulable=True,
        require_carryable=True,
        require_stateful=False,
        require_style=require_style,
    )
    rng.shuffle(candidates)
    for candidate in candidates:
        location_with_preposition = _sample_feasible_location(scene, candidate, rng)
        if location_with_preposition is None:
            continue
        location, preposition = location_with_preposition
        instruction = (
            f"{verb} {object_phrase.format(object_class=format_object_name(candidate.object_class))} "
            f"{preposition} the {location.description}"
        )
        return GeneratedInstruction(instruction=instruction, grammar_rule=grammar_rule)
    return None


def _generate_state_change(
    scene: Phase2Output,
    rng: random.Random,
    *,
    grammar_rule: str,
    action: str,
    required_state: str,
    object_phrase: str,
    require_style: bool = False,
) -> GeneratedInstruction | None:
    candidate = sample_one(
        _eligible_state_candidates(scene, required_state, require_style=require_style),
        rng,
    )
    if candidate is None:
        return None
    instruction = f"{action} {object_phrase.format(object_class=format_object_name(candidate.object_class))}"
    return GeneratedInstruction(instruction=instruction, grammar_rule=grammar_rule)


def _generate_handover_private_like_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_private_object(
        scene, rng, grammar_rule="handover_private_like_object", suffix="like"
    )


def _generate_handover_private_hate_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_private_object(
        scene, rng, grammar_rule="handover_private_hate_object", suffix="hate"
    )


def _generate_handover_private_prefer_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_private_object(
        scene, rng, grammar_rule="handover_private_prefer_object", suffix="prefer"
    )


def _generate_handover_private_favorite_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_private_object(
        scene,
        rng,
        grammar_rule="handover_private_favorite_object",
        suffix="favorite",
        possessive=True,
    )


def _generate_handover_private_least_favorite_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_private_object(
        scene,
        rng,
        grammar_rule="handover_private_least_favorite_object",
        suffix="least favorite",
        possessive=True,
    )


def _generate_handover_most_stylish_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_descriptor_object(
        scene,
        rng,
        grammar_rule="handover_most_stylish_object",
        descriptor="most stylish",
        require_style=True,
    )


def _generate_handover_best_looking_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_descriptor_object(
        scene, rng, grammar_rule="handover_best_looking_object", descriptor="best-looking"
    )


def _generate_handover_nicest_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_descriptor_object(
        scene, rng, grammar_rule="handover_nicest_object", descriptor="nicest"
    )


def _generate_handover_prettiest_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_descriptor_object(
        scene, rng, grammar_rule="handover_prettiest_object", descriptor="prettiest"
    )


def _generate_handover_ugliest_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_descriptor_object(
        scene, rng, grammar_rule="handover_ugliest_object", descriptor="ugliest"
    )


def _generate_handover_coolest_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_descriptor_object(
        scene, rng, grammar_rule="handover_coolest_object", descriptor="coolest"
    )


def _generate_handover_most_appealing_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_descriptor_object(
        scene,
        rng,
        grammar_rule="handover_most_appealing_object",
        descriptor="most appealing",
    )


def _generate_handover_least_appealing_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_handover_descriptor_object(
        scene,
        rng,
        grammar_rule="handover_least_appealing_object",
        descriptor="least appealing",
    )


def _generate_put_private_like_object_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_placement(
        scene,
        rng,
        grammar_rule="put_private_like_object_location",
        verb="put",
        object_phrase="the {object_class} I like",
    )


def _generate_place_private_prefer_object_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_placement(
        scene,
        rng,
        grammar_rule="place_private_prefer_object_location",
        verb="place",
        object_phrase="the {object_class} I prefer",
    )


def _generate_put_most_stylish_object_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_placement(
        scene,
        rng,
        grammar_rule="put_most_stylish_object_location",
        verb="put",
        object_phrase="the most stylish {object_class}",
        require_style=True,
    )


def _generate_place_best_looking_object_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_placement(
        scene,
        rng,
        grammar_rule="place_best_looking_object_location",
        verb="place",
        object_phrase="the best-looking {object_class}",
    )


def _generate_open_private_like_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_change(
        scene,
        rng,
        grammar_rule="open_private_like_object",
        action="open",
        required_state="closed",
        object_phrase="the {object_class} I like",
    )


def _generate_close_private_prefer_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_change(
        scene,
        rng,
        grammar_rule="close_private_prefer_object",
        action="close",
        required_state="open",
        object_phrase="the {object_class} I prefer",
    )


def _generate_open_private_favorite_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_change(
        scene,
        rng,
        grammar_rule="open_private_favorite_object",
        action="open",
        required_state="closed",
        object_phrase="my favorite {object_class}",
    )


def _generate_close_nicer_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_change(
        scene,
        rng,
        grammar_rule="close_nicer_object",
        action="close",
        required_state="open",
        object_phrase="the nicer {object_class}",
    )


def _generate_turn_on_best_looking_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_change(
        scene,
        rng,
        grammar_rule="turn_on_best_looking_object",
        action="turn on",
        required_state="off",
        object_phrase="the best-looking {object_class}",
    )


def _generate_turn_off_most_stylish_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_state_change(
        scene,
        rng,
        grammar_rule="turn_off_most_stylish_object",
        action="turn off",
        required_state="on",
        object_phrase="the most stylish {object_class}",
        require_style=True,
    )


RULE_GENERATORS = {
    "handover_private_like_object": _generate_handover_private_like_object,
    "handover_private_hate_object": _generate_handover_private_hate_object,
    "handover_private_prefer_object": _generate_handover_private_prefer_object,
    "handover_private_favorite_object": _generate_handover_private_favorite_object,
    "handover_private_least_favorite_object": _generate_handover_private_least_favorite_object,
    "handover_most_stylish_object": _generate_handover_most_stylish_object,
    "handover_best_looking_object": _generate_handover_best_looking_object,
    "handover_nicest_object": _generate_handover_nicest_object,
    "handover_prettiest_object": _generate_handover_prettiest_object,
    "handover_ugliest_object": _generate_handover_ugliest_object,
    "handover_coolest_object": _generate_handover_coolest_object,
    "handover_most_appealing_object": _generate_handover_most_appealing_object,
    "handover_least_appealing_object": _generate_handover_least_appealing_object,
    "put_private_like_object_location": _generate_put_private_like_object_location,
    "place_private_prefer_object_location": _generate_place_private_prefer_object_location,
    "put_most_stylish_object_location": _generate_put_most_stylish_object_location,
    "place_best_looking_object_location": _generate_place_best_looking_object_location,
    "open_private_like_object": _generate_open_private_like_object,
    "close_private_prefer_object": _generate_close_private_prefer_object,
    "open_private_favorite_object": _generate_open_private_favorite_object,
    "close_nicer_object": _generate_close_nicer_object,
    "turn_on_best_looking_object": _generate_turn_on_best_looking_object,
    "turn_off_most_stylish_object": _generate_turn_off_most_stylish_object,
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
