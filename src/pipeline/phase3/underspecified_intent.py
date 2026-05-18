from __future__ import annotations

import random
from dataclasses import asdict

from pipeline.phase1.vocabs import LOCATION_TYPE_VOCAB, STATE_VOCAB
from pipeline.phase2.models import (
    Phase2Output,
    UnderspecifiedLocationCandidate,
    UnderspecifiedObjectCandidate,
)
from pipeline.phase3.helper import (
    GeneratedInstruction,
    GrammarRule,
    format_object_name,
    sample_one,
)


REQUIRED_STATES = {"open", "closed", "on", "off", "full", "empty"}
if not REQUIRED_STATES.issubset(STATE_VOCAB):
    missing_states = sorted(REQUIRED_STATES - set(STATE_VOCAB))
    raise RuntimeError(f"STATE_VOCAB is missing required underspecified intent states: {missing_states}")

REQUIRED_LOCATION_TYPES = {
    "surface",
    "shelf",
    "floor_region",
    "hanging_point",
    "container",
    "drawer",
    "inside_container",
}
if not REQUIRED_LOCATION_TYPES.issubset(LOCATION_TYPE_VOCAB):
    missing_location_types = sorted(REQUIRED_LOCATION_TYPES - set(LOCATION_TYPE_VOCAB))
    raise RuntimeError(
        "LOCATION_TYPE_VOCAB is missing required underspecified intent "
        f"location types: {missing_location_types}"
    )


GRAMMAR = """
<underspecified_intent> ::= <object_only_underspecified>
                          | <state_change_underspecified>
                          | <object_underspecified_location_explicit>
                          | <object_explicit_location_underspecified>
                          | <object_and_location_underspecified>

<object_only_underspecified> ::= <pick_it_up>
                               | <bring_it_here>
                               | <give_it_to_me>
                               | <put_it_down>
                               | <move_that_out_of_the_way>

<state_change_underspecified> ::= <open_it>
                                | <close_it>
                                | <turn_it_on>
                                | <turn_it_off>
                                | <empty_it>
                                | <fill_it>

<object_underspecified_location_explicit> ::= <put_it_on_location>
                                            | <place_it_on_location>
                                            | <set_it_on_location>
                                            | <put_it_in_location>
                                            | <place_it_in_location>
                                            | <put_it_inside_location>
                                            | <set_that_on_location>

<object_explicit_location_underspecified> ::= <put_object_there>
                                            | <place_object_over_there>
                                            | <move_object_there>
                                            | <bring_object_here>
                                            | <set_object_down_there>

<object_and_location_underspecified> ::= <put_it_there>
                                       | <place_it_there>
                                       | <move_it_there>
                                       | <bring_it_here_with_destination>
                                       | <take_it_over_there>
                                       | <put_that_there>
                                       | <place_that_here>

<pick_it_up> ::= "pick it up"
<bring_it_here> ::= "bring it here"
<give_it_to_me> ::= "give it to me"
<put_it_down> ::= "put it down"
<move_that_out_of_the_way> ::= "move that out of the way"

<open_it> ::= "open it"
<close_it> ::= "close it"
<turn_it_on> ::= "turn it on"
<turn_it_off> ::= "turn it off"
<empty_it> ::= "empty it"
<fill_it> ::= "fill it"

<put_it_on_location> ::= "put it on the " <explicit_on_location>
<place_it_on_location> ::= "place it on the " <explicit_on_location>
<set_it_on_location> ::= "set it on the " <explicit_on_location>
<put_it_in_location> ::= "put it in the " <explicit_in_location>
<place_it_in_location> ::= "place it in the " <explicit_in_location>
<put_it_inside_location> ::= "put it inside the " <explicit_in_location>
<set_that_on_location> ::= "set that on the " <explicit_on_location>

<put_object_there> ::= "put the " <explicit_manipulable_object> " there"
<place_object_over_there> ::= "place the " <explicit_manipulable_object> " over there"
<move_object_there> ::= "move the " <explicit_manipulable_object> " there"
<bring_object_here> ::= "bring the " <explicit_manipulable_object> " here"
<set_object_down_there> ::= "set the " <explicit_manipulable_object> " down there"

<put_it_there> ::= "put it there"
<place_it_there> ::= "place it there"
<move_it_there> ::= "move it there"
<bring_it_here_with_destination> ::= "bring it here"
<take_it_over_there> ::= "take it over there"
<put_that_there> ::= "put that there"
<place_that_here> ::= "place that here"

<implicit_manipulable_object> ::= an underspecified_object_candidate with is_manipulable=true and exceeds_weight_limit=false
<implicit_stateful_object> ::= an underspecified_object_candidate with is_stateful=true and a compatible state
<explicit_manipulable_object> ::= an underspecified_object_candidate.object_class with is_manipulable=true and exceeds_weight_limit=false
<explicit_on_location> ::= an underspecified_location_candidate.description whose location_type is surface, shelf, or floor_region
<explicit_in_location> ::= an underspecified_location_candidate.description whose location_type is container, drawer, or inside_container
""".strip()


RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        name="pick_it_up",
        pattern="pick it up",
        slots=(),
        notes="Requires at least one carryable underspecified_object_candidate.",
    ),
    GrammarRule(
        name="bring_it_here",
        pattern="bring it here",
        slots=(),
        notes="Requires at least one carryable underspecified_object_candidate.",
    ),
    GrammarRule(
        name="give_it_to_me",
        pattern="give it to me",
        slots=(),
        notes="Requires at least one carryable underspecified_object_candidate.",
    ),
    GrammarRule(
        name="put_it_down",
        pattern="put it down",
        slots=(),
        notes="Requires at least one carryable underspecified_object_candidate.",
    ),
    GrammarRule(
        name="move_that_out_of_the_way",
        pattern="move that out of the way",
        slots=(),
        notes="Requires at least one carryable underspecified_object_candidate.",
    ),
    GrammarRule(
        name="open_it",
        pattern="open it",
        slots=(),
        notes='Requires an underspecified_object_candidate with is_stateful=true and state="closed".',
    ),
    GrammarRule(
        name="close_it",
        pattern="close it",
        slots=(),
        notes='Requires an underspecified_object_candidate with is_stateful=true and state="open".',
    ),
    GrammarRule(
        name="turn_it_on",
        pattern="turn it on",
        slots=(),
        notes='Requires an underspecified_object_candidate with is_stateful=true and state="off".',
    ),
    GrammarRule(
        name="turn_it_off",
        pattern="turn it off",
        slots=(),
        notes='Requires an underspecified_object_candidate with is_stateful=true and state="on".',
    ),
    GrammarRule(
        name="empty_it",
        pattern="empty it",
        slots=(),
        notes='Requires an underspecified_object_candidate with is_stateful=true and state="full".',
    ),
    GrammarRule(
        name="fill_it",
        pattern="fill it",
        slots=(),
        notes='Requires an underspecified_object_candidate with is_stateful=true and state="empty".',
    ),
    GrammarRule(
        name="put_it_on_location",
        pattern="put it on the {location}",
        slots=("location",),
        notes="Requires a carryable object candidate and an explicit on-location candidate.",
    ),
    GrammarRule(
        name="place_it_on_location",
        pattern="place it on the {location}",
        slots=("location",),
        notes="Requires a carryable object candidate and an explicit on-location candidate.",
    ),
    GrammarRule(
        name="set_it_on_location",
        pattern="set it on the {location}",
        slots=("location",),
        notes="Requires a carryable object candidate and an explicit on-location candidate.",
    ),
    GrammarRule(
        name="put_it_in_location",
        pattern="put it in the {location}",
        slots=("location",),
        notes="Requires a carryable object candidate and an explicit in-location candidate.",
    ),
    GrammarRule(
        name="place_it_in_location",
        pattern="place it in the {location}",
        slots=("location",),
        notes="Requires a carryable object candidate and an explicit in-location candidate.",
    ),
    GrammarRule(
        name="put_it_inside_location",
        pattern="put it inside the {location}",
        slots=("location",),
        notes="Requires a carryable object candidate and an explicit in-location candidate.",
    ),
    GrammarRule(
        name="set_that_on_location",
        pattern="set that on the {location}",
        slots=("location",),
        notes="Requires a carryable object candidate and an explicit on-location candidate.",
    ),
    GrammarRule(
        name="put_object_there",
        pattern="put the {object_class} there",
        slots=("object_class",),
        notes="Requires an explicit carryable object candidate.",
    ),
    GrammarRule(
        name="place_object_over_there",
        pattern="place the {object_class} over there",
        slots=("object_class",),
        notes="Requires an explicit carryable object candidate.",
    ),
    GrammarRule(
        name="move_object_there",
        pattern="move the {object_class} there",
        slots=("object_class",),
        notes="Requires an explicit carryable object candidate.",
    ),
    GrammarRule(
        name="bring_object_here",
        pattern="bring the {object_class} here",
        slots=("object_class",),
        notes="Requires an explicit carryable object candidate.",
    ),
    GrammarRule(
        name="set_object_down_there",
        pattern="set the {object_class} down there",
        slots=("object_class",),
        notes="Requires an explicit carryable object candidate.",
    ),
    GrammarRule(
        name="put_it_there",
        pattern="put it there",
        slots=(),
        notes="Requires at least one carryable object candidate and one location candidate.",
    ),
    GrammarRule(
        name="place_it_there",
        pattern="place it there",
        slots=(),
        notes="Requires at least one carryable object candidate and one location candidate.",
    ),
    GrammarRule(
        name="move_it_there",
        pattern="move it there",
        slots=(),
        notes="Requires at least one carryable object candidate and one location candidate.",
    ),
    GrammarRule(
        name="bring_it_here_with_destination",
        pattern="bring it here",
        slots=(),
        notes="Requires at least one carryable object candidate and one location candidate.",
    ),
    GrammarRule(
        name="take_it_over_there",
        pattern="take it over there",
        slots=(),
        notes="Requires at least one carryable object candidate and one location candidate.",
    ),
    GrammarRule(
        name="put_that_there",
        pattern="put that there",
        slots=(),
        notes="Requires at least one carryable object candidate and one location candidate.",
    ),
    GrammarRule(
        name="place_that_here",
        pattern="place that here",
        slots=(),
        notes="Requires at least one carryable object candidate and one location candidate.",
    ),
)


def get_grammar() -> str:
    return GRAMMAR


def get_rules() -> tuple[GrammarRule, ...]:
    return RULES


ON_LOCATION_TYPES = {"surface", "shelf", "floor_region"}
IN_LOCATION_TYPES = {"container", "drawer", "inside_container"}


def _carryable_objects(scene: Phase2Output) -> list[UnderspecifiedObjectCandidate]:
    return [
        candidate
        for candidate in scene.checks.underspecified_object_candidates
        if candidate.is_manipulable and not candidate.exceeds_weight_limit
    ]


def _stateful_objects(
    scene: Phase2Output,
    state: str,
) -> list[UnderspecifiedObjectCandidate]:
    return [
        candidate
        for candidate in scene.checks.underspecified_object_candidates
        if candidate.is_stateful and candidate.state == state
    ]


def _locations(
    scene: Phase2Output,
    allowed_types: set[str] | None = None,
) -> list[UnderspecifiedLocationCandidate]:
    if allowed_types is None:
        return list(scene.checks.underspecified_location_candidates)
    return [
        candidate
        for candidate in scene.checks.underspecified_location_candidates
        if candidate.location_type in allowed_types
    ]


def _has_carryable_object(scene: Phase2Output) -> bool:
    return bool(_carryable_objects(scene))


def _has_location(scene: Phase2Output) -> bool:
    return bool(scene.checks.underspecified_location_candidates)


def _generate_constant_instruction(
    scene: Phase2Output,
    grammar_rule: str,
    instruction: str,
    *,
    require_location: bool = False,
) -> GeneratedInstruction | None:
    if not _has_carryable_object(scene):
        return None
    if require_location and not _has_location(scene):
        return None
    return GeneratedInstruction(instruction=instruction, grammar_rule=grammar_rule)


def _generate_pick_it_up(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_constant_instruction(scene, "pick_it_up", "pick it up")


def _generate_bring_it_here(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_constant_instruction(scene, "bring_it_here", "bring it here")


def _generate_give_it_to_me(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_constant_instruction(scene, "give_it_to_me", "give it to me")


def _generate_put_it_down(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_constant_instruction(scene, "put_it_down", "put it down")


def _generate_move_that_out_of_the_way(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_constant_instruction(
        scene,
        "move_that_out_of_the_way",
        "move that out of the way",
    )


def _generate_state_instruction(
    scene: Phase2Output,
    grammar_rule: str,
    instruction: str,
    required_state: str,
) -> GeneratedInstruction | None:
    if not _stateful_objects(scene, required_state):
        return None
    return GeneratedInstruction(instruction=instruction, grammar_rule=grammar_rule)


def _generate_open_it(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_state_instruction(scene, "open_it", "open it", "closed")


def _generate_close_it(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_state_instruction(scene, "close_it", "close it", "open")


def _generate_turn_it_on(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_state_instruction(scene, "turn_it_on", "turn it on", "off")


def _generate_turn_it_off(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_state_instruction(scene, "turn_it_off", "turn it off", "on")


def _generate_empty_it(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_state_instruction(scene, "empty_it", "empty it", "full")


def _generate_fill_it(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_state_instruction(scene, "fill_it", "fill it", "empty")


def _generate_location_instruction(
    scene: Phase2Output,
    rng: random.Random,
    grammar_rule: str,
    pattern: str,
    allowed_types: set[str],
) -> GeneratedInstruction | None:
    if not _has_carryable_object(scene):
        return None
    location = sample_one(_locations(scene, allowed_types), rng)
    if location is None:
        return None
    return GeneratedInstruction(
        instruction=pattern.format(location=location.description),
        grammar_rule=grammar_rule,
    )


def _generate_put_it_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_instruction(
        scene, rng, "put_it_on_location", "put it on the {location}", ON_LOCATION_TYPES
    )


def _generate_place_it_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_instruction(
        scene, rng, "place_it_on_location", "place it on the {location}", ON_LOCATION_TYPES
    )


def _generate_set_it_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_instruction(
        scene, rng, "set_it_on_location", "set it on the {location}", ON_LOCATION_TYPES
    )


def _generate_put_it_in_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_instruction(
        scene, rng, "put_it_in_location", "put it in the {location}", IN_LOCATION_TYPES
    )


def _generate_place_it_in_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_instruction(
        scene, rng, "place_it_in_location", "place it in the {location}", IN_LOCATION_TYPES
    )


def _generate_put_it_inside_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_instruction(
        scene,
        rng,
        "put_it_inside_location",
        "put it inside the {location}",
        IN_LOCATION_TYPES,
    )


def _generate_set_that_on_location(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_location_instruction(
        scene, rng, "set_that_on_location", "set that on the {location}", ON_LOCATION_TYPES
    )


def _generate_object_instruction(
    scene: Phase2Output,
    rng: random.Random,
    grammar_rule: str,
    pattern: str,
) -> GeneratedInstruction | None:
    if not _has_location(scene):
        return None
    obj = sample_one(_carryable_objects(scene), rng)
    if obj is None:
        return None
    return GeneratedInstruction(
        instruction=pattern.format(object_class=format_object_name(obj.object_class)),
        grammar_rule=grammar_rule,
    )


def _generate_put_object_there(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_instruction(
        scene, rng, "put_object_there", "put the {object_class} there"
    )


def _generate_place_object_over_there(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_instruction(
        scene, rng, "place_object_over_there", "place the {object_class} over there"
    )


def _generate_move_object_there(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_instruction(
        scene, rng, "move_object_there", "move the {object_class} there"
    )


def _generate_bring_object_here(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_instruction(
        scene, rng, "bring_object_here", "bring the {object_class} here"
    )


def _generate_set_object_down_there(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_object_instruction(
        scene, rng, "set_object_down_there", "set the {object_class} down there"
    )


def _generate_put_it_there(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_constant_instruction(
        scene, "put_it_there", "put it there", require_location=True
    )


def _generate_place_it_there(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_constant_instruction(
        scene, "place_it_there", "place it there", require_location=True
    )


def _generate_move_it_there(scene: Phase2Output, rng: random.Random) -> GeneratedInstruction | None:
    return _generate_constant_instruction(
        scene, "move_it_there", "move it there", require_location=True
    )


def _generate_bring_it_here_with_destination(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_constant_instruction(
        scene, "bring_it_here_with_destination", "bring it here", require_location=True
    )


def _generate_take_it_over_there(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_constant_instruction(
        scene, "take_it_over_there", "take it over there", require_location=True
    )


def _generate_put_that_there(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_constant_instruction(
        scene, "put_that_there", "put that there", require_location=True
    )


def _generate_place_that_here(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    return _generate_constant_instruction(
        scene, "place_that_here", "place that here", require_location=True
    )


RULE_GENERATORS = {
    "pick_it_up": _generate_pick_it_up,
    "bring_it_here": _generate_bring_it_here,
    "give_it_to_me": _generate_give_it_to_me,
    "put_it_down": _generate_put_it_down,
    "move_that_out_of_the_way": _generate_move_that_out_of_the_way,
    "open_it": _generate_open_it,
    "close_it": _generate_close_it,
    "turn_it_on": _generate_turn_it_on,
    "turn_it_off": _generate_turn_it_off,
    "empty_it": _generate_empty_it,
    "fill_it": _generate_fill_it,
    "put_it_on_location": _generate_put_it_on_location,
    "place_it_on_location": _generate_place_it_on_location,
    "set_it_on_location": _generate_set_it_on_location,
    "put_it_in_location": _generate_put_it_in_location,
    "place_it_in_location": _generate_place_it_in_location,
    "put_it_inside_location": _generate_put_it_inside_location,
    "set_that_on_location": _generate_set_that_on_location,
    "put_object_there": _generate_put_object_there,
    "place_object_over_there": _generate_place_object_over_there,
    "move_object_there": _generate_move_object_there,
    "bring_object_here": _generate_bring_object_here,
    "set_object_down_there": _generate_set_object_down_there,
    "put_it_there": _generate_put_it_there,
    "place_it_there": _generate_place_it_there,
    "move_it_there": _generate_move_it_there,
    "bring_it_here_with_destination": _generate_bring_it_here_with_destination,
    "take_it_over_there": _generate_take_it_over_there,
    "put_that_there": _generate_put_that_there,
    "place_that_here": _generate_place_that_here,
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
