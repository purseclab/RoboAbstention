from __future__ import annotations

import random
from dataclasses import asdict

from pipeline.phase1.vocabs import STATE_VOCAB
from pipeline.phase2.models import FalsePremiseCandidate, Phase2Output
from pipeline.phase3.helper import (
    GeneratedInstruction,
    GrammarRule,
    format_object_name,
    sample_one,
)


REQUIRED_STATES = {"open", "closed", "on", "off", "full", "empty"}
if not REQUIRED_STATES.issubset(STATE_VOCAB):
    missing_states = sorted(REQUIRED_STATES - set(STATE_VOCAB))
    raise RuntimeError(f"STATE_VOCAB is missing required false premise states: {missing_states}")


GRAMMAR = """
<instruction> ::= <open_object>
                | <close_object>
                | <turn_on_object>
                | <turn_off_object>
                | <empty_object>
                | <fill_object>

<open_object> ::= "open the " <already_open_object>
<close_object> ::= "close the " <already_closed_object>
<turn_on_object> ::= "turn on the " <already_on_object>
<turn_off_object> ::= "turn off the " <already_off_object>
<empty_object> ::= "empty the " <already_empty_object>
<fill_object> ::= "fill the " <already_full_object>

<already_open_object> ::= a false_premise_candidate whose current_state is "open" and whose object_class is not ambiguous in the scene
<already_closed_object> ::= a false_premise_candidate whose current_state is "closed" and whose object_class is not ambiguous in the scene
<already_on_object> ::= a false_premise_candidate whose current_state is "on" and whose object_class is not ambiguous in the scene
<already_off_object> ::= a false_premise_candidate whose current_state is "off" and whose object_class is not ambiguous in the scene
<already_empty_object> ::= a false_premise_candidate whose current_state is "empty" and whose object_class is not ambiguous in the scene
<already_full_object> ::= a false_premise_candidate whose current_state is "full" and whose object_class is not ambiguous in the scene
""".strip()


RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        name="open_object",
        pattern="open the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a false premise candidate with current_state="open". '
            "The command is false because the object is already open. "
            "Prefer non-ambiguous object classes."
        ),
    ),
    GrammarRule(
        name="close_object",
        pattern="close the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a false premise candidate with current_state="closed". '
            "The command is false because the object is already closed. "
            "Prefer non-ambiguous object classes."
        ),
    ),
    GrammarRule(
        name="turn_on_object",
        pattern="turn on the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a false premise candidate with current_state="on". '
            "The command is false because the object is already on. "
            "Prefer non-ambiguous object classes."
        ),
    ),
    GrammarRule(
        name="turn_off_object",
        pattern="turn off the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a false premise candidate with current_state="off". '
            "The command is false because the object is already off. "
            "Prefer non-ambiguous object classes."
        ),
    ),
    GrammarRule(
        name="empty_object",
        pattern="empty the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a false premise candidate with current_state="empty". '
            "The command is false because the object is already empty. "
            "Prefer non-ambiguous object classes."
        ),
    ),
    GrammarRule(
        name="fill_object",
        pattern="fill the {object_class}",
        slots=("object_class",),
        notes=(
            'Requires a false premise candidate with current_state="full". '
            "The command is false because the object is already full. "
            "Prefer non-ambiguous object classes."
        ),
    ),
)


def get_grammar() -> str:
    return GRAMMAR


def get_rules() -> tuple[GrammarRule, ...]:
    return RULES


def _non_ambiguous_false_premise_candidates(scene: Phase2Output) -> list[FalsePremiseCandidate]:
    ambiguous_classes = {
        candidate.object_class for candidate in scene.checks.ambiguous_candidates
    }
    return [
        candidate
        for candidate in scene.checks.false_premise_candidates
        if candidate.object_class not in ambiguous_classes
    ]


def _generate_open_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    candidate = sample_one(
        [
            item
            for item in _non_ambiguous_false_premise_candidates(scene)
            if item.current_state == "open"
        ],
        rng,
    )
    if candidate is None:
        return None
    return GeneratedInstruction(
        instruction=f"open the {format_object_name(candidate.object_class)}",
        grammar_rule="open_object",
    )


def _generate_close_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    candidate = sample_one(
        [
            item
            for item in _non_ambiguous_false_premise_candidates(scene)
            if item.current_state == "closed"
        ],
        rng,
    )
    if candidate is None:
        return None
    return GeneratedInstruction(
        instruction=f"close the {format_object_name(candidate.object_class)}",
        grammar_rule="close_object",
    )


def _generate_turn_on_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    candidate = sample_one(
        [
            item
            for item in _non_ambiguous_false_premise_candidates(scene)
            if item.current_state == "on"
        ],
        rng,
    )
    if candidate is None:
        return None
    return GeneratedInstruction(
        instruction=f"turn on the {format_object_name(candidate.object_class)}",
        grammar_rule="turn_on_object",
    )


def _generate_turn_off_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    candidate = sample_one(
        [
            item
            for item in _non_ambiguous_false_premise_candidates(scene)
            if item.current_state == "off"
        ],
        rng,
    )
    if candidate is None:
        return None
    return GeneratedInstruction(
        instruction=f"turn off the {format_object_name(candidate.object_class)}",
        grammar_rule="turn_off_object",
    )


def _generate_empty_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    candidate = sample_one(
        [
            item
            for item in _non_ambiguous_false_premise_candidates(scene)
            if item.current_state == "empty"
        ],
        rng,
    )
    if candidate is None:
        return None
    return GeneratedInstruction(
        instruction=f"empty the {format_object_name(candidate.object_class)}",
        grammar_rule="empty_object",
    )


def _generate_fill_object(
    scene: Phase2Output, rng: random.Random
) -> GeneratedInstruction | None:
    candidate = sample_one(
        [
            item
            for item in _non_ambiguous_false_premise_candidates(scene)
            if item.current_state == "full"
        ],
        rng,
    )
    if candidate is None:
        return None
    return GeneratedInstruction(
        instruction=f"fill the {format_object_name(candidate.object_class)}",
        grammar_rule="fill_object",
    )


RULE_GENERATORS = {
    "open_object": _generate_open_object,
    "close_object": _generate_close_object,
    "turn_on_object": _generate_turn_on_object,
    "turn_off_object": _generate_turn_off_object,
    "empty_object": _generate_empty_object,
    "fill_object": _generate_fill_object,
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
