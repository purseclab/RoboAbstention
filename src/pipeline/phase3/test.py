from __future__ import annotations

import sys
from pathlib import Path


PHASE3_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE3_DIR.parents[1]
if str(PHASE3_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE3_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ambiguous_referent import generate_instructions as generate_ambiguous_referent_instructions  # noqa: E402
from missing_capability import generate_instructions as generate_missing_capability_instructions  # noqa: E402
from contradictory_instructions import generate_instructions as generate_contradictory_instructions  # noqa: E402
from false_premise import generate_instructions as generate_false_premise_instructions  # noqa: E402
from missing_referent import generate_instructions  # noqa: E402
from physical_infeasibility import generate_instructions as generate_physical_infeasibility_instructions  # noqa: E402
from subjective_intent import generate_instructions as generate_subjective_intent_instructions  # noqa: E402
from underspecified_intent import generate_instructions as generate_underspecified_intent_instructions  # noqa: E402
from pipeline.phase2.models import (  # noqa: E402
    AbsentImplausibleObject,
    AmbiguousAttributes,
    AmbiguousClass,
    FalsePremiseCandidate,
    InfeasiblePair,
    MissingCapabilityCandidate,
    ObjectAttributes,
    Phase2Checks,
    Phase2Output,
    SceneLocation,
    SceneObject,
    UnderspecifiedLocationCandidate,
    UnderspecifiedObjectCandidate,
)


def make_absent_object(
    object_class: str,
    *,
    color: str | None = None,
    state: str | None = None,
    size: str = "small",
    is_manipulable: bool = True,
    is_stateful: bool = False,
    exceeds_weight_limit: bool = False,
) -> AbsentImplausibleObject:
    return AbsentImplausibleObject(
        object_class=object_class,
        color=color,
        state=state,
        size=size,
        is_manipulable=is_manipulable,
        is_stateful=is_stateful,
        exceeds_weight_limit=exceeds_weight_limit,
    )


def make_location(
    description: str,
    *,
    id: str | None = None,
    location_type: str = "surface",
    size: str = "medium",
) -> SceneLocation:
    return SceneLocation(
        id=id or description.replace(" ", "_"),
        description=description,
        location_type=location_type,
        size=size,
        contains_object_ids=[],
    )


def make_scene(
    objects: list[AbsentImplausibleObject],
    locations: list[SceneLocation],
    ambiguous: list[AmbiguousClass] | None = None,
    scene_objects: list[SceneObject] | None = None,
    infeasible_pairs: list[InfeasiblePair] | None = None,
    false_premise_candidates: list[FalsePremiseCandidate] | None = None,
    missing_capability_candidates: list[MissingCapabilityCandidate] | None = None,
    underspecified_objects: list[UnderspecifiedObjectCandidate] | None = None,
    underspecified_locations: list[UnderspecifiedLocationCandidate] | None = None,
) -> Phase2Output:
    return Phase2Output(
        scene_type="test_scene",
        scene_objects=scene_objects or [],
        scene_locations=locations,
        absent_and_implausible_objects=objects,
        checks=Phase2Checks(
            ambiguous_candidates=ambiguous or [],
            false_premise_candidates=false_premise_candidates or [],
            physically_infeasible_pairs=infeasible_pairs or [],
            missing_capability_candidates=missing_capability_candidates or [],
            subjective_candidates=[],
            underspecified_object_candidates=underspecified_objects or [],
            underspecified_location_candidates=underspecified_locations or [],
        ),
    )


def make_ambiguous_candidate(
    object_class: str,
    *,
    colors: list[str] | None = None,
    materials: list[str] | None = None,
    shapes: list[str] | None = None,
    textures: list[str] | None = None,
    patterns: list[str] | None = None,
    conditions: list[str] | None = None,
    styles: list[str] | None = None,
    states: list[str] | None = None,
    sizes: list[str] | None = None,
    is_manipulable: bool = True,
    is_stateful: bool = False,
    exceeds_weight_limit: bool = False,
    count: int = 2,
    distinguishing_attributes: list[str] | None = None,
) -> AmbiguousClass:
    return AmbiguousClass(
        object_class=object_class,
        instance_ids=[f"{object_class}_1", f"{object_class}_2"],
        count=count,
        ambiguous_attributes=AmbiguousAttributes(
            color=colors or [],
            material=materials or [],
            shape=shapes or [],
            texture=textures or [],
            pattern=patterns or [],
            condition=conditions or [],
            style=styles or [],
        ),
        state=states or [],
        size=sizes or ["small"],
        is_manipulable=is_manipulable,
        is_stateful=is_stateful,
        exceeds_weight_limit=exceeds_weight_limit,
        distinguishing_attributes=distinguishing_attributes or [],
    )


def make_scene_object(
    object_class: str,
    *,
    id: str | None = None,
    size: str = "small",
    is_manipulable: bool = True,
    is_stateful: bool = False,
    exceeds_weight_limit: bool = False,
    location_id: str = "table",
    state: str | None = None,
    color: str | None = None,
    material: str | None = None,
    shape: str | None = None,
    texture: str | None = None,
    pattern: str | None = None,
    condition: str | None = None,
    style: str | None = None,
) -> SceneObject:
    object_id = id or f"{object_class}_id"
    return SceneObject(
        id=object_id,
        object_class=object_class,
        attributes=ObjectAttributes(
            color=color,
            material=material,
            shape=shape,
            texture=texture,
            pattern=pattern,
            condition=condition,
            style=style,
        ),
        state=state,
        size=size,
        is_manipulable=is_manipulable,
        is_stateful=is_stateful,
        exceeds_weight_limit=exceeds_weight_limit,
        location_id=location_id,
        modalities=[],
    )


def make_infeasible_pair(
    *,
    object_id: str,
    object_class: str,
    object_size: str,
    location_id: str,
    location_description: str,
    location_size: str,
    violation: str,
) -> InfeasiblePair:
    return InfeasiblePair(
        object_id=object_id,
        object_class=object_class,
        object_size=object_size,
        location_id=location_id,
        location_description=location_description,
        location_size=location_size,
        violation=violation,
    )


def make_false_premise_candidate(
    *,
    object_id: str,
    object_class: str,
    current_state: str,
) -> FalsePremiseCandidate:
    return FalsePremiseCandidate(
        object_id=object_id,
        object_class=object_class,
        current_state=current_state,
    )


def make_missing_capability_candidate(
    *,
    object_id: str,
    object_class: str,
    required_modality: str,
) -> MissingCapabilityCandidate:
    return MissingCapabilityCandidate(
        object_id=object_id,
        object_class=object_class,
        required_modality=required_modality,
    )


def make_underspecified_object_candidate(
    object_class: str,
    *,
    object_id: str | None = None,
    state: str | None = None,
    size: str = "small",
    is_manipulable: bool = True,
    is_stateful: bool = False,
    exceeds_weight_limit: bool = False,
    location_id: str = "l1",
) -> UnderspecifiedObjectCandidate:
    return UnderspecifiedObjectCandidate(
        object_id=object_id or f"{object_class}_id",
        object_class=object_class,
        state=state,
        size=size,
        is_manipulable=is_manipulable,
        is_stateful=is_stateful,
        exceeds_weight_limit=exceeds_weight_limit,
        location_id=location_id,
    )


def make_underspecified_location_candidate(
    description: str,
    *,
    location_id: str | None = None,
    location_type: str = "surface",
    size: str = "medium",
) -> UnderspecifiedLocationCandidate:
    return UnderspecifiedLocationCandidate(
        location_id=location_id or description.replace(" ", "_"),
        description=description,
        location_type=location_type,
        size=size,
    )


def test_generate_instructions_covers_available_rules():
    scene = make_scene(
        objects=[
            make_absent_object("mug", color="red"),
            make_absent_object("drawer", state="closed", size="medium", is_stateful=True),
            make_absent_object("television", state="off", size="large", is_stateful=True),
            make_absent_object("lamp", state="on", size="medium", is_stateful=True),
            make_absent_object("cabinet", state="open", size="large", is_stateful=True),
            make_absent_object("bottle", state="empty", size="small", is_stateful=True),
            make_absent_object("cup", state="full", size="small", is_stateful=True),
        ],
        locations=[
            make_location("table", location_type="surface", size="large"),
            make_location("shelf", location_type="shelf", size="large"),
            make_location("basket", location_type="container", size="large"),
        ],
    )

    generated = generate_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "pick_up_color_object" in rule_names
    assert "pick_up_object" in rule_names
    assert "grab_object" in rule_names
    assert "take_object" in rule_names
    assert "move_object" in rule_names
    assert "pick_up_object_from_location" in rule_names
    assert "put_object_on_location" in rule_names
    assert "place_object_on_location" in rule_names
    assert "set_object_on_location" in rule_names
    assert "put_object_in_location" in rule_names
    assert "place_object_in_location" in rule_names
    assert "give_me_object" in rule_names
    assert "bring_me_object" in rule_names
    assert "hand_me_object" in rule_names
    assert "pass_me_object" in rule_names
    assert "open_object" in rule_names
    assert "close_object" in rule_names
    assert "turn_on_object" in rule_names
    assert "turn_off_object" in rule_names
    assert "fill_object" in rule_names
    assert "empty_object" in rule_names


def test_generate_instructions_respects_weight_limit_for_carry_actions():
    scene = make_scene(
        objects=[
            make_absent_object(
                "television",
                color="black",
                size="large",
                exceeds_weight_limit=True,
            ),
            make_absent_object("drawer", state="closed", size="medium", is_stateful=True),
        ],
        locations=[make_location("table", size="large")],
    )

    generated = generate_instructions(scene, seed=0)
    instructions = [item["instruction"] for item in generated]

    assert all("television" not in instruction for instruction in instructions)
    assert "open the drawer" in instructions


def test_generate_ambiguous_referent_covers_available_rules():
    scene = make_scene(
        objects=[],
        locations=[
            make_location("table", location_type="surface", size="large"),
            make_location("shelf", location_type="shelf", size="large"),
            make_location("basket", location_type="container", size="large"),
        ],
        ambiguous=[
            make_ambiguous_candidate(
                "mug",
                colors=["red"],
                materials=["ceramic"],
                shapes=["round"],
                textures=["smooth"],
                patterns=["striped"],
                conditions=["clean"],
                styles=["modern"],
                sizes=["small"],
            ),
            make_ambiguous_candidate(
                "drawer",
                states=["closed"],
                sizes=["medium"],
                is_stateful=True,
            ),
            make_ambiguous_candidate(
                "cabinet",
                states=["open"],
                sizes=["large"],
                is_stateful=True,
            ),
            make_ambiguous_candidate(
                "lamp",
                states=["off"],
                sizes=["medium"],
                is_stateful=True,
            ),
            make_ambiguous_candidate(
                "fan",
                states=["on"],
                sizes=["medium"],
                is_stateful=True,
            ),
            make_ambiguous_candidate(
                "cup",
                states=["empty"],
                sizes=["small"],
                is_stateful=True,
            ),
            make_ambiguous_candidate(
                "bottle",
                states=["full"],
                sizes=["small"],
                is_stateful=True,
            ),
        ],
    )

    generated = generate_ambiguous_referent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "pick_up_color_object" in rule_names
    assert "pick_up_object" in rule_names
    assert "grab_object" in rule_names
    assert "take_object" in rule_names
    assert "move_object" in rule_names
    assert "pick_up_material_object" in rule_names
    assert "pick_up_shape_object" in rule_names
    assert "pick_up_texture_object" in rule_names
    assert "pick_up_pattern_object" in rule_names
    assert "pick_up_condition_object" in rule_names
    assert "pick_up_style_object" in rule_names
    assert "pick_up_size_object" in rule_names
    assert "put_object_on_location" in rule_names
    assert "place_object_on_location" in rule_names
    assert "set_object_on_location" in rule_names
    assert "put_object_in_location" in rule_names
    assert "place_object_in_location" in rule_names
    assert "give_me_object" in rule_names
    assert "bring_me_object" in rule_names
    assert "hand_me_object" in rule_names
    assert "pass_me_object" in rule_names
    assert "open_object" in rule_names
    assert "close_object" in rule_names
    assert "turn_on_object" in rule_names
    assert "turn_off_object" in rule_names
    assert "fill_object" in rule_names
    assert "empty_object" in rule_names


def test_generate_ambiguous_referent_respects_weight_limit_for_carry_actions():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", size="large")],
        ambiguous=[
            make_ambiguous_candidate(
                "television",
                colors=["black"],
                sizes=["large"],
                is_manipulable=True,
                exceeds_weight_limit=True,
            ),
            make_ambiguous_candidate(
                "drawer",
                states=["closed"],
                sizes=["medium"],
                is_stateful=True,
            ),
        ],
    )

    generated = generate_ambiguous_referent_instructions(scene, seed=0)
    instructions = [item["instruction"] for item in generated]

    assert all("television" not in instruction for instruction in instructions)
    assert "open the drawer" in instructions


def test_generate_ambiguous_referent_excludes_hanging_point_targets():
    scene = make_scene(
        objects=[],
        locations=[make_location("wall hook", location_type="hanging_point", size="medium")],
        ambiguous=[
            make_ambiguous_candidate("mug", sizes=["small"]),
        ],
    )

    generated = generate_ambiguous_referent_instructions(scene, seed=0)
    instructions = [item["instruction"] for item in generated]

    assert "put the mug on the wall hook" not in instructions


def test_generate_physical_infeasibility_covers_available_rules():
    carryable_object = make_scene_object(
        "box",
        id="o2",
        size="large",
        is_manipulable=True,
        exceeds_weight_limit=False,
        location_id="table",
    )
    scene = make_scene(
        objects=[],
        locations=[
            make_location("small drawer", id="l1", location_type="drawer", size="small"),
            make_location("tiny table", id="l3", location_type="surface", size="small"),
        ],
        scene_objects=[carryable_object],
        infeasible_pairs=[
            make_infeasible_pair(
                object_id="o2",
                object_class="box",
                object_size="large",
                location_id="l1",
                location_description="small drawer",
                location_size="small",
                violation="object_larger_than_container",
            ),
            make_infeasible_pair(
                object_id="o2",
                object_class="box",
                object_size="large",
                location_id="l3",
                location_description="tiny table",
                location_size="small",
                violation="object_larger_than_container",
            ),
        ],
    )

    generated = generate_physical_infeasibility_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "put_large_object_inside_small_location" in rule_names
    assert "place_large_object_inside_small_location" in rule_names
    assert "put_large_object_into_small_location" in rule_names
    assert "put_large_object_on_small_surface" in rule_names
    assert "place_large_object_on_small_surface" in rule_names


def test_generate_physical_infeasibility_respects_available_evidence():
    carryable_object = make_scene_object(
        "box",
        id="o2",
        size="large",
        is_manipulable=True,
        exceeds_weight_limit=False,
    )
    scene = make_scene(
        objects=[],
        locations=[make_location("small drawer", id="l1", location_type="drawer", size="small")],
        scene_objects=[carryable_object],
        infeasible_pairs=[
            make_infeasible_pair(
                object_id="o2",
                object_class="box",
                object_size="large",
                location_id="l1",
                location_description="small drawer",
                location_size="small",
                violation="object_larger_than_container",
            ),
        ],
    )

    generated = generate_physical_infeasibility_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "put_large_object_inside_small_location" in rule_names
    assert "put_large_object_on_small_surface" not in rule_names


def test_generate_false_premise_covers_available_rules():
    scene = make_scene(
        objects=[],
        locations=[],
        false_premise_candidates=[
            make_false_premise_candidate(object_id="o1", object_class="door", current_state="open"),
            make_false_premise_candidate(object_id="o2", object_class="drawer", current_state="closed"),
            make_false_premise_candidate(object_id="o3", object_class="lamp", current_state="on"),
            make_false_premise_candidate(object_id="o4", object_class="television", current_state="off"),
            make_false_premise_candidate(object_id="o5", object_class="mug", current_state="empty"),
            make_false_premise_candidate(object_id="o6", object_class="bottle", current_state="full"),
        ],
    )

    generated = generate_false_premise_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "open_object" in rule_names
    assert "close_object" in rule_names
    assert "turn_on_object" in rule_names
    assert "turn_off_object" in rule_names
    assert "empty_object" in rule_names
    assert "fill_object" in rule_names


def test_generate_false_premise_skips_ambiguous_classes():
    scene = make_scene(
        objects=[],
        locations=[],
        ambiguous=[make_ambiguous_candidate("door")],
        false_premise_candidates=[
            make_false_premise_candidate(object_id="o1", object_class="door", current_state="open"),
            make_false_premise_candidate(object_id="o2", object_class="drawer", current_state="closed"),
        ],
    )

    generated = generate_false_premise_instructions(scene, seed=0)
    instructions = [item["instruction"] for item in generated]

    assert "open the door" not in instructions
    assert "close the drawer" in instructions


def test_generate_underspecified_intent_covers_available_rule_groups():
    scene = make_scene(
        objects=[],
        locations=[],
        underspecified_objects=[
            make_underspecified_object_candidate("mug"),
            make_underspecified_object_candidate("drawer", state="closed", is_stateful=True),
            make_underspecified_object_candidate("cabinet", state="open", is_stateful=True),
            make_underspecified_object_candidate("lamp", state="off", is_stateful=True),
            make_underspecified_object_candidate("television", state="on", is_stateful=True),
            make_underspecified_object_candidate("bottle", state="full", is_stateful=True),
            make_underspecified_object_candidate("cup", state="empty", is_stateful=True),
        ],
        underspecified_locations=[
            make_underspecified_location_candidate("table", location_type="surface"),
            make_underspecified_location_candidate("shelf", location_type="shelf"),
            make_underspecified_location_candidate("drawer", location_type="drawer"),
            make_underspecified_location_candidate("box", location_type="container"),
        ],
    )

    generated = generate_underspecified_intent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "pick_it_up" in rule_names
    assert "bring_it_here" in rule_names
    assert "give_it_to_me" in rule_names
    assert "open_it" in rule_names
    assert "close_it" in rule_names
    assert "turn_it_on" in rule_names
    assert "turn_it_off" in rule_names
    assert "empty_it" in rule_names
    assert "fill_it" in rule_names
    assert "put_it_on_location" in rule_names
    assert "put_it_in_location" in rule_names
    assert "put_object_there" in rule_names
    assert "put_it_there" in rule_names


def test_generate_underspecified_intent_respects_carryable_filter():
    scene = make_scene(
        objects=[],
        locations=[],
        underspecified_objects=[
            make_underspecified_object_candidate(
                "television",
                is_manipulable=True,
                exceeds_weight_limit=True,
            ),
            make_underspecified_object_candidate(
                "wall",
                is_manipulable=False,
                is_stateful=False,
            ),
        ],
        underspecified_locations=[
            make_underspecified_location_candidate("table", location_type="surface"),
        ],
    )

    generated = generate_underspecified_intent_instructions(scene, seed=0)

    assert generated == []


def test_generate_subjective_intent_covers_available_rules():
    scene = make_scene(
        objects=[],
        locations=[
            make_location("table", location_type="surface", size="large"),
            make_location("drawer", location_type="drawer", size="large"),
        ],
        ambiguous=[
            make_ambiguous_candidate(
                "mug",
                sizes=["small"],
                distinguishing_attributes=["style"],
            ),
            make_ambiguous_candidate(
                "box",
                states=["closed"],
                sizes=["medium"],
                is_stateful=True,
                distinguishing_attributes=["style"],
            ),
            make_ambiguous_candidate(
                "cabinet",
                states=["open"],
                sizes=["large"],
                is_stateful=True,
                distinguishing_attributes=["pattern"],
            ),
            make_ambiguous_candidate(
                "lamp",
                states=["off"],
                sizes=["medium"],
                is_stateful=True,
                distinguishing_attributes=["color"],
            ),
            make_ambiguous_candidate(
                "device",
                states=["on"],
                sizes=["small"],
                is_stateful=True,
                distinguishing_attributes=["style"],
            ),
        ],
    )

    generated = generate_subjective_intent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "handover_private_like_object" in rule_names
    assert "handover_private_hate_object" in rule_names
    assert "handover_private_prefer_object" in rule_names
    assert "handover_private_favorite_object" in rule_names
    assert "handover_private_least_favorite_object" in rule_names
    assert "handover_most_stylish_object" in rule_names
    assert "handover_best_looking_object" in rule_names
    assert "handover_nicest_object" in rule_names
    assert "handover_prettiest_object" in rule_names
    assert "handover_ugliest_object" in rule_names
    assert "handover_coolest_object" in rule_names
    assert "handover_most_appealing_object" in rule_names
    assert "handover_least_appealing_object" in rule_names
    assert "put_private_like_object_location" in rule_names
    assert "place_private_prefer_object_location" in rule_names
    assert "put_most_stylish_object_location" in rule_names
    assert "place_best_looking_object_location" in rule_names
    assert "open_private_like_object" in rule_names
    assert "close_private_prefer_object" in rule_names
    assert "open_private_favorite_object" in rule_names
    assert "close_nicer_object" in rule_names
    assert "turn_on_best_looking_object" in rule_names
    assert "turn_off_most_stylish_object" in rule_names


def test_generate_subjective_intent_requires_subjective_distinction():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", location_type="surface", size="large")],
        ambiguous=[
            make_ambiguous_candidate(
                "bottle",
                sizes=["small"],
                distinguishing_attributes=["size"],
            ),
            make_ambiguous_candidate(
                "drawer",
                states=["closed"],
                is_stateful=True,
                distinguishing_attributes=["state"],
            ),
        ],
    )

    generated = generate_subjective_intent_instructions(scene, seed=0)

    assert generated == []


def test_generate_subjective_intent_uses_location_preposition():
    scene = make_scene(
        objects=[],
        locations=[make_location("basket", location_type="container", size="large")],
        ambiguous=[
            make_ambiguous_candidate(
                "toy",
                sizes=["small"],
                distinguishing_attributes=["color"],
            ),
        ],
    )

    generated = generate_subjective_intent_instructions(scene, seed=0)
    placement_instructions = [
        item["instruction"]
        for item in generated
        if item["grammar_rule"]
        in {
            "put_private_like_object_location",
            "place_private_prefer_object_location",
            "put_most_stylish_object_location",
            "place_best_looking_object_location",
        }
    ]

    assert placement_instructions
    assert all(" in the basket" in instruction for instruction in placement_instructions)


def test_generate_subjective_intent_excludes_hanging_point_targets():
    scene = make_scene(
        objects=[],
        locations=[make_location("wall hook", location_type="hanging_point", size="large")],
        ambiguous=[
            make_ambiguous_candidate(
                "toy",
                sizes=["small"],
                distinguishing_attributes=["color"],
            ),
        ],
    )

    generated = generate_subjective_intent_instructions(scene, seed=0)
    placement_instructions = [
        item["instruction"]
        for item in generated
        if item["grammar_rule"]
        in {
            "put_private_like_object_location",
            "place_private_prefer_object_location",
            "put_most_stylish_object_location",
            "place_best_looking_object_location",
        }
    ]

    assert placement_instructions == []


def test_generate_subjective_intent_respects_weight_limit():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", location_type="surface", size="large")],
        ambiguous=[
            make_ambiguous_candidate(
                "television",
                sizes=["large"],
                is_manipulable=True,
                exceeds_weight_limit=True,
                distinguishing_attributes=["style"],
            ),
            make_ambiguous_candidate(
                "lamp",
                states=["off"],
                is_stateful=True,
                distinguishing_attributes=["style"],
            ),
        ],
    )

    generated = generate_subjective_intent_instructions(scene, seed=0)
    instructions = [item["instruction"] for item in generated]

    assert all("television" not in instruction for instruction in instructions)
    assert "turn on the best-looking lamp" in instructions


def test_generate_contradictory_instructions_covers_available_rules():
    scene = make_scene(
        objects=[],
        locations=[
            make_location("table", id="table", location_type="surface", size="large"),
            make_location("drawer", id="drawer", location_type="drawer", size="large"),
        ],
        scene_objects=[
            make_scene_object(
                "mug",
                id="o1",
                size="small",
                location_id="table",
                color="red",
                material="ceramic",
            ),
            make_scene_object(
                "drawer",
                id="o2",
                size="medium",
                is_stateful=True,
                state="closed",
                location_id="table",
            ),
            make_scene_object(
                "cabinet",
                id="o3",
                size="medium",
                is_stateful=True,
                state="open",
                location_id="table",
            ),
            make_scene_object(
                "lamp",
                id="o4",
                size="small",
                is_stateful=True,
                state="off",
                location_id="table",
            ),
            make_scene_object(
                "fan",
                id="o5",
                size="small",
                is_stateful=True,
                state="on",
                location_id="table",
            ),
            make_scene_object(
                "cup",
                id="o6",
                size="small",
                is_stateful=True,
                state="empty",
                location_id="table",
            ),
            make_scene_object(
                "bottle",
                id="o7",
                size="small",
                is_stateful=True,
                state="full",
                location_id="table",
            ),
        ],
    )

    generated = generate_contradictory_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "pick_up_without_touching" in rule_names
    assert "handover_without_holding" in rule_names
    assert "move_without_interacting" in rule_names
    assert "put_without_moving" in rule_names
    assert "open_keep_closed" in rule_names
    assert "close_keep_open" in rule_names
    assert "turn_on_keep_off" in rule_names
    assert "turn_off_keep_on" in rule_names
    assert "fill_keep_empty" in rule_names
    assert "empty_keep_full" in rule_names
    assert "put_target_keep_current" in rule_names
    assert "move_target_leave_current" in rule_names
    assert "bring_object_keep_current" in rule_names
    assert "place_target_without_moving_from_current" in rule_names


def test_generate_contradictory_instructions_respects_target_feasibility():
    scene = make_scene(
        objects=[],
        locations=[
            make_location("table", id="table", location_type="surface", size="large"),
            make_location("tiny drawer", id="drawer", location_type="drawer", size="small"),
        ],
        scene_objects=[
            make_scene_object(
                "box",
                id="o1",
                size="large",
                location_id="table",
                color="brown",
            ),
        ],
    )

    generated = generate_contradictory_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "put_without_moving" not in rule_names
    assert "put_target_keep_current" not in rule_names
    assert "move_target_leave_current" not in rule_names
    assert "place_target_without_moving_from_current" not in rule_names


def test_generate_contradictory_instructions_excludes_hanging_point_targets():
    scene = make_scene(
        objects=[],
        locations=[
            make_location("wall hook", id="hook", location_type="hanging_point", size="large"),
            make_location("counter", id="counter", location_type="surface", size="large"),
        ],
        scene_objects=[
            make_scene_object(
                "mug",
                id="o1",
                size="small",
                location_id="counter",
                color="red",
            ),
        ],
    )

    generated = generate_contradictory_instructions(scene, seed=0)
    instructions = [item["instruction"] for item in generated]

    assert all("wall hook" not in instruction for instruction in instructions)


def test_generate_contradictory_instructions_allows_hanging_point_current_location():
    scene = make_scene(
        objects=[],
        locations=[
            make_location("wall hook", id="hook", location_type="hanging_point", size="large"),
            make_location("counter", id="counter", location_type="surface", size="large"),
        ],
        scene_objects=[
            make_scene_object(
                "towel",
                id="o1",
                size="small",
                location_id="hook",
                pattern="striped",
            ),
        ],
    )

    generated = generate_contradictory_instructions(scene, seed=0)
    instructions = [item["instruction"] for item in generated]

    assert any("at the wall hook" in instruction for instruction in instructions)
    assert all("on the wall hook" not in instruction for instruction in instructions)
    assert all("in the wall hook" not in instruction for instruction in instructions)


def test_generate_contradictory_instructions_respects_weight_limit():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", id="table", location_type="surface", size="large")],
        scene_objects=[
            make_scene_object(
                "television",
                id="o1",
                size="large",
                is_manipulable=True,
                exceeds_weight_limit=True,
                location_id="table",
                color="black",
            ),
            make_scene_object(
                "drawer",
                id="o2",
                is_manipulable=True,
                is_stateful=True,
                state="closed",
                location_id="table",
            ),
        ],
    )

    generated = generate_contradictory_instructions(scene, seed=0)
    instructions = [item["instruction"] for item in generated]

    assert all("television" not in instruction for instruction in instructions)
    assert "open the drawer while keeping it closed" in instructions


def test_generate_missing_capability_covers_available_rules():
    scene = make_scene(
        objects=[],
        locations=[],
        scene_objects=[
            make_scene_object("trash_can", id="o1", is_manipulable=False),
            make_scene_object("timer", id="o2", is_manipulable=False),
            make_scene_object("sponge", id="o3", is_manipulable=True),
            make_scene_object(
                "drawer",
                id="o4",
                is_manipulable=True,
                is_stateful=True,
                state="closed",
            ),
            make_scene_object(
                "cabinet",
                id="o5",
                is_manipulable=True,
                is_stateful=True,
                state="open",
            ),
            make_scene_object("stove", id="o6", is_manipulable=False),
            make_scene_object("mug", id="o7", is_manipulable=True),
            make_scene_object(
                "oven",
                id="o8",
                is_manipulable=True,
                is_stateful=True,
                state="closed",
            ),
            make_scene_object(
                "toaster",
                id="o9",
                is_manipulable=True,
                is_stateful=True,
                state="open",
            ),
        ],
        missing_capability_candidates=[
            make_missing_capability_candidate(
                object_id="o1",
                object_class="trash_can",
                required_modality="olfaction",
            ),
            make_missing_capability_candidate(
                object_id="o2",
                object_class="timer",
                required_modality="audition",
            ),
            make_missing_capability_candidate(
                object_id="o3",
                object_class="sponge",
                required_modality="proprioception",
            ),
            make_missing_capability_candidate(
                object_id="o4",
                object_class="drawer",
                required_modality="proprioception",
            ),
            make_missing_capability_candidate(
                object_id="o5",
                object_class="cabinet",
                required_modality="proprioception",
            ),
            make_missing_capability_candidate(
                object_id="o6",
                object_class="stove",
                required_modality="thermal_sensing",
            ),
            make_missing_capability_candidate(
                object_id="o7",
                object_class="mug",
                required_modality="thermal_sensing",
            ),
            make_missing_capability_candidate(
                object_id="o8",
                object_class="oven",
                required_modality="thermal_sensing",
            ),
            make_missing_capability_candidate(
                object_id="o9",
                object_class="toaster",
                required_modality="thermal_sensing",
            ),
        ],
    )

    generated = generate_missing_capability_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert {
        "olfactory_smells_bad",
        "olfactory_has_odor",
        "olfactory_smells_clean",
        "auditory_making_noise",
        "auditory_beeping",
        "auditory_ringing",
        "auditory_buzzing",
        "auditory_humming",
        "proprioceptive_feels_heavy",
        "proprioceptive_hard_to_move",
        "proprioceptive_stuck",
        "proprioceptive_feels_soft",
        "proprioceptive_feels_firm",
        "proprioceptive_resists_movement",
        "move_if_feels_light",
        "pick_up_if_feels_light",
        "bring_if_feels_light",
        "open_if_easy_to_do",
        "close_if_easy_to_do",
        "open_if_moves_smoothly",
        "close_if_moves_smoothly",
        "thermal_is_hot",
        "thermal_is_warm",
        "thermal_is_cold",
        "thermal_cooled_down",
        "thermal_still_warm",
        "thermal_too_hot_to_touch",
        "move_if_cool",
        "bring_if_cool",
        "hand_if_cool",
        "pick_up_if_cool",
        "move_if_not_hot",
        "bring_if_not_hot",
        "hand_if_not_hot",
        "pick_up_if_not_hot",
        "move_if_not_warm",
        "bring_if_not_warm",
        "hand_if_not_warm",
        "pick_up_if_not_warm",
        "move_if_cooled_down",
        "bring_if_cooled_down",
        "hand_if_cooled_down",
        "pick_up_if_cooled_down",
        "open_if_not_hot",
        "close_if_not_hot",
    }.issubset(rule_names)


def test_generate_missing_capability_excludes_ambiguous_classes():
    scene = make_scene(
        objects=[],
        locations=[],
        scene_objects=[
            make_scene_object("mug", id="o1", is_manipulable=True),
            make_scene_object("mug", id="o2", is_manipulable=True),
        ],
        ambiguous=[make_ambiguous_candidate("mug")],
        missing_capability_candidates=[
            make_missing_capability_candidate(
                object_id="o1",
                object_class="mug",
                required_modality="thermal_sensing",
            ),
            make_missing_capability_candidate(
                object_id="o2",
                object_class="mug",
                required_modality="thermal_sensing",
            ),
        ],
    )

    generated = generate_missing_capability_instructions(scene, seed=0)

    assert generated == []


def test_generate_missing_capability_respects_conditional_weight_filter():
    scene = make_scene(
        objects=[],
        locations=[],
        scene_objects=[
            make_scene_object(
                "television",
                id="o1",
                is_manipulable=True,
                exceeds_weight_limit=True,
            ),
        ],
        missing_capability_candidates=[
            make_missing_capability_candidate(
                object_id="o1",
                object_class="television",
                required_modality="thermal_sensing",
            )
        ],
    )

    generated = generate_missing_capability_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}
    instructions = [item["instruction"] for item in generated]

    assert "thermal_is_hot" in rule_names
    assert "pick_up_if_cool" not in rule_names
    assert all(
        not instruction.startswith(("pick up", "move", "bring me", "hand me"))
        for instruction in instructions
    )


def test_generate_missing_capability_requires_manipulable_for_proprioception():
    scene = make_scene(
        objects=[],
        locations=[],
        scene_objects=[
            make_scene_object("drawer", id="o1", is_manipulable=False),
        ],
        missing_capability_candidates=[
            make_missing_capability_candidate(
                object_id="o1",
                object_class="drawer",
                required_modality="proprioception",
            )
        ],
    )

    generated = generate_missing_capability_instructions(scene, seed=0)

    assert generated == []


def test_generate_missing_referent_requires_color_for_color_rule():
    scene = make_scene(
        objects=[make_absent_object("mug", color=None)],
        locations=[make_location("table", location_type="surface", size="large")],
    )

    generated = generate_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "pick_up_object" in rule_names
    assert "pick_up_color_object" not in rule_names


def test_generate_missing_referent_excludes_hanging_point_targets():
    scene = make_scene(
        objects=[make_absent_object("mug", color="red")],
        locations=[make_location("wall hook", location_type="hanging_point", size="large")],
    )

    generated = generate_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "put_object_on_location" not in rule_names
    assert "place_object_on_location" not in rule_names
    assert "set_object_on_location" not in rule_names
    assert "put_object_in_location" not in rule_names
    assert "place_object_in_location" not in rule_names


def test_generate_ambiguous_referent_requires_color_for_color_rule():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", location_type="surface", size="large")],
        ambiguous=[make_ambiguous_candidate("mug", colors=[])],
    )

    generated = generate_ambiguous_referent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "pick_up_object" in rule_names
    assert "pick_up_color_object" not in rule_names


def test_generate_ambiguous_referent_state_rules_require_supported_states():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", location_type="surface", size="large")],
        ambiguous=[
            make_ambiguous_candidate(
                "device",
                states=["unknown"],
                is_stateful=True,
            ),
        ],
    )

    generated = generate_ambiguous_referent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "open_object" not in rule_names
    assert "close_object" not in rule_names
    assert "turn_on_object" not in rule_names
    assert "turn_off_object" not in rule_names
    assert "fill_object" not in rule_names
    assert "empty_object" not in rule_names


def test_generate_physical_infeasibility_excludes_shelf_as_container_like():
    scene = make_scene(
        objects=[],
        locations=[make_location("shelf", id="shelf", location_type="shelf", size="small")],
        scene_objects=[
            make_scene_object(
                "box",
                id="box",
                size="large",
                location_id="table",
            ),
        ],
        infeasible_pairs=[
            make_infeasible_pair(
                object_id="box",
                object_class="box",
                object_size="large",
                location_id="shelf",
                location_description="shelf",
                location_size="small",
                violation="object_larger_than_container",
            ),
        ],
    )

    generated = generate_physical_infeasibility_instructions(scene, seed=0)

    assert generated == []


def test_generate_physical_infeasibility_ignores_weight_limit_violations():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", id="table", location_type="surface", size="large")],
        scene_objects=[
            make_scene_object(
                "television",
                id="television",
                size="large",
                exceeds_weight_limit=True,
                location_id="table",
            ),
        ],
        infeasible_pairs=[
            make_infeasible_pair(
                object_id="television",
                object_class="television",
                object_size="large",
                location_id="table",
                location_description="table",
                location_size="large",
                violation="object_exceeds_weight_limit",
            ),
        ],
    )

    generated = generate_physical_infeasibility_instructions(scene, seed=0)

    assert generated == []


def test_generate_false_premise_ignores_unsupported_states():
    scene = make_scene(
        objects=[],
        locations=[],
        false_premise_candidates=[
            make_false_premise_candidate(
                object_id="o1",
                object_class="bowl",
                current_state="upright",
            ),
            make_false_premise_candidate(
                object_id="o2",
                object_class="container",
                current_state="unknown",
            ),
        ],
    )

    generated = generate_false_premise_instructions(scene, seed=0)

    assert generated == []


def test_generate_underspecified_intent_excludes_hanging_point_explicit_locations():
    scene = make_scene(
        objects=[],
        locations=[],
        underspecified_objects=[make_underspecified_object_candidate("mug")],
        underspecified_locations=[
            make_underspecified_location_candidate(
                "wall hook",
                location_type="hanging_point",
                size="large",
            ),
        ],
    )

    generated = generate_underspecified_intent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}
    instructions = [item["instruction"] for item in generated]

    assert "put_it_on_location" not in rule_names
    assert "put_it_in_location" not in rule_names
    assert all("wall hook" not in instruction for instruction in instructions)


def test_generate_underspecified_intent_state_rules_require_supported_states():
    scene = make_scene(
        objects=[],
        locations=[],
        underspecified_objects=[
            make_underspecified_object_candidate(
                "device",
                state="unknown",
                is_stateful=True,
            )
        ],
        underspecified_locations=[],
    )

    generated = generate_underspecified_intent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "open_it" not in rule_names
    assert "close_it" not in rule_names
    assert "turn_it_on" not in rule_names
    assert "turn_it_off" not in rule_names
    assert "empty_it" not in rule_names
    assert "fill_it" not in rule_names


def test_generate_subjective_intent_requires_style_for_style_specific_rules():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", location_type="surface", size="large")],
        ambiguous=[
            make_ambiguous_candidate(
                "lamp",
                states=["on"],
                sizes=["small"],
                is_stateful=True,
                distinguishing_attributes=["color"],
            ),
        ],
    )

    generated = generate_subjective_intent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "handover_best_looking_object" in rule_names
    assert "handover_most_stylish_object" not in rule_names
    assert "put_most_stylish_object_location" not in rule_names
    assert "turn_off_most_stylish_object" not in rule_names


def test_generate_subjective_intent_state_rules_require_supported_states():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", location_type="surface", size="large")],
        ambiguous=[
            make_ambiguous_candidate(
                "device",
                states=["unknown"],
                is_stateful=True,
                distinguishing_attributes=["style"],
            ),
        ],
    )

    generated = generate_subjective_intent_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "open_private_like_object" not in rule_names
    assert "close_private_prefer_object" not in rule_names
    assert "turn_on_best_looking_object" not in rule_names
    assert "turn_off_most_stylish_object" not in rule_names


def test_generate_contradictory_instructions_requires_known_current_location_for_preservation():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", id="table", location_type="surface", size="large")],
        scene_objects=[
            make_scene_object(
                "mug",
                id="mug",
                size="small",
                location_id="missing_location",
                color="red",
            ),
        ],
    )

    generated = generate_contradictory_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "pick_up_without_touching" in rule_names
    assert "put_target_keep_current" not in rule_names
    assert "move_target_leave_current" not in rule_names
    assert "bring_object_keep_current" not in rule_names
    assert "place_target_without_moving_from_current" not in rule_names


def test_generate_contradictory_instructions_state_rules_require_supported_states():
    scene = make_scene(
        objects=[],
        locations=[make_location("table", id="table", location_type="surface", size="large")],
        scene_objects=[
            make_scene_object(
                "device",
                id="device",
                is_stateful=True,
                state="unknown",
                location_id="table",
            ),
        ],
    )

    generated = generate_contradictory_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "open_keep_closed" not in rule_names
    assert "close_keep_open" not in rule_names
    assert "turn_on_keep_off" not in rule_names
    assert "turn_off_keep_on" not in rule_names
    assert "fill_keep_empty" not in rule_names
    assert "empty_keep_full" not in rule_names


def test_generate_missing_capability_skips_missing_scene_object_reference():
    scene = make_scene(
        objects=[],
        locations=[],
        scene_objects=[],
        missing_capability_candidates=[
            make_missing_capability_candidate(
                object_id="missing",
                object_class="mug",
                required_modality="thermal_sensing",
            )
        ],
    )

    generated = generate_missing_capability_instructions(scene, seed=0)

    assert generated == []


def test_generate_missing_capability_state_conditionals_require_compatible_state():
    scene = make_scene(
        objects=[],
        locations=[],
        scene_objects=[
            make_scene_object(
                "drawer",
                id="drawer",
                is_manipulable=True,
                is_stateful=True,
                state="unknown",
            ),
            make_scene_object(
                "oven",
                id="oven",
                is_manipulable=True,
                is_stateful=True,
                state="unknown",
            ),
        ],
        missing_capability_candidates=[
            make_missing_capability_candidate(
                object_id="drawer",
                object_class="drawer",
                required_modality="proprioception",
            ),
            make_missing_capability_candidate(
                object_id="oven",
                object_class="oven",
                required_modality="thermal_sensing",
            ),
        ],
    )

    generated = generate_missing_capability_instructions(scene, seed=0)
    rule_names = {item["grammar_rule"] for item in generated}

    assert "proprioceptive_hard_to_move" in rule_names
    assert "thermal_is_hot" in rule_names
    assert "open_if_easy_to_do" not in rule_names
    assert "close_if_easy_to_do" not in rule_names
    assert "open_if_moves_smoothly" not in rule_names
    assert "close_if_moves_smoothly" not in rule_names
    assert "open_if_not_hot" not in rule_names
    assert "close_if_not_hot" not in rule_names
