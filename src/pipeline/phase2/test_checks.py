from __future__ import annotations

import sys
from pathlib import Path

import pytest


PHASE2_DIR = Path(__file__).resolve().parent
if str(PHASE2_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE2_DIR))

from checks import (  # noqa: E402
    check_ambiguous,
    check_false_premise,
    check_missing_capability,
    check_physically_infeasible,
    check_subjective,
    check_underspecified_locations,
    check_underspecified_objects,
)
from models import (  # noqa: E402
    ObjectAttributes,
    Phase1Output,
    SceneLocation,
    SceneObject,
)


def build_scene(
    objects: list[SceneObject] | None = None,
    locations: list[SceneLocation] | None = None,
) -> Phase1Output:
    return Phase1Output(
        scene_type="test scene",
        scene_objects=objects or [],
        scene_locations=locations or [],
        absent_and_implausible_objects=[],
    )


def make_object(
    id: str = "o1",
    object_class: str = "cup",
    color: str | None = None,
    material: str | None = None,
    shape: str | None = None,
    texture: str | None = None,
    pattern: str | None = None,
    condition: str | None = None,
    style: str | None = None,
    state: str | None = None,
    size: str = "small",
    is_manipulable: bool = True,
    is_stateful: bool = False,
    location_id: str = "l1",
    modalities: list[str] | None = None,
    exceeds_weight_limit: bool = False,
) -> SceneObject:
    return SceneObject(
        id=id,
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
        modalities=modalities or [],
    )


def make_location(
    id: str = "l1",
    description: str = "table surface",
    location_type: str = "surface",
    size: str = "medium",
    contains_object_ids: list[str] | None = None,
) -> SceneLocation:
    return SceneLocation(
        id=id,
        description=description,
        location_type=location_type,
        size=size,
        contains_object_ids=contains_object_ids or [],
    )


def test_ambiguous_two_identical_objects():
    scene = build_scene(
        objects=[
            make_object(
                id="o1",
                object_class="chair",
                color="black",
                material="fabric",
                shape="rectangular",
            ),
            make_object(
                id="o2",
                object_class="chair",
                color="black",
                material="fabric",
                shape="rectangular",
            ),
        ]
    )

    result = check_ambiguous(scene)

    assert len(result) == 1
    candidate = result[0]
    assert candidate.instance_ids == ["o1", "o2"]
    assert candidate.count == 2
    assert candidate.ambiguous_attributes.color == ["black"]
    assert candidate.ambiguous_attributes.material == ["fabric"]
    assert candidate.ambiguous_attributes.shape == ["rectangular"]
    assert candidate.ambiguous_attributes.pattern == []
    assert candidate.ambiguous_attributes.condition == []
    assert candidate.ambiguous_attributes.style == []
    assert candidate.state == []
    assert candidate.size == ["small"]
    assert candidate.is_manipulable is True
    assert candidate.is_stateful is False
    assert candidate.exceeds_weight_limit is False
    assert candidate.distinguishing_attributes == []


def test_ambiguous_two_objects_one_differing_attribute():
    scene = build_scene(
        objects=[
            make_object(
                id="o1",
                object_class="chair",
                color="black",
                material="fabric",
                shape="rectangular",
            ),
            make_object(
                id="o2",
                object_class="chair",
                color="brown",
                material="fabric",
                shape="rectangular",
            ),
        ]
    )

    result = check_ambiguous(scene)

    assert len(result) == 1
    candidate = result[0]
    assert candidate.ambiguous_attributes.color == []
    assert candidate.size == ["small"]
    assert candidate.distinguishing_attributes == ["color"]


def test_ambiguous_three_objects_partial_overlap():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="chair", color="black", material="mesh"),
            make_object(id="o2", object_class="chair", color="black", material="cushion"),
            make_object(id="o3", object_class="chair", color="brown", material="mesh"),
        ]
    )

    result = check_ambiguous(scene)

    assert len(result) == 1
    candidate = result[0]
    assert candidate.ambiguous_attributes.color == ["black"]
    assert candidate.ambiguous_attributes.material == ["mesh"]
    assert candidate.size == ["small"]
    assert candidate.distinguishing_attributes == []


def test_ambiguous_repeated_subjective_attributes():
    scene = build_scene(
        objects=[
            make_object(
                id="o1",
                object_class="bowl",
                pattern="striped",
                condition="clean",
                style="decorative",
            ),
            make_object(
                id="o2",
                object_class="bowl",
                pattern="striped",
                condition="clean",
                style="simple",
            ),
            make_object(
                id="o3",
                object_class="bowl",
                pattern="solid",
                condition="worn",
                style="decorative",
            ),
        ]
    )

    result = check_ambiguous(scene)

    assert len(result) == 1
    candidate = result[0]
    assert candidate.ambiguous_attributes.pattern == ["striped"]
    assert candidate.ambiguous_attributes.condition == ["clean"]
    assert candidate.ambiguous_attributes.style == ["decorative"]
    assert candidate.distinguishing_attributes == []


def test_ambiguous_single_instance():
    scene = build_scene(objects=[make_object(object_class="chair")])

    result = check_ambiguous(scene)

    assert result == []


def test_ambiguous_different_classes():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="chair"),
            make_object(id="o2", object_class="table"),
        ]
    )

    result = check_ambiguous(scene)

    assert result == []


def test_ambiguous_null_not_distinguishing():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="chair", color=None),
            make_object(id="o2", object_class="chair", color="black"),
        ]
    )

    result = check_ambiguous(scene)

    assert len(result) == 1
    candidate = result[0]
    assert candidate.ambiguous_attributes.color == []
    assert candidate.distinguishing_attributes == []


def test_ambiguous_multiple_classes():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="chair"),
            make_object(id="o2", object_class="chair"),
            make_object(id="o3", object_class="cup"),
            make_object(id="o4", object_class="cup"),
        ]
    )

    result = check_ambiguous(scene)

    assert len(result) == 2
    assert sorted(candidate.object_class for candidate in result) == ["chair", "cup"]


def test_ambiguous_repeated_state_and_size():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="drawer", state="closed", is_stateful=True, size="medium"),
            make_object(id="o2", object_class="drawer", state="closed", is_stateful=True, size="medium"),
            make_object(id="o3", object_class="drawer", state="open", is_stateful=True, size="large"),
        ]
    )

    result = check_ambiguous(scene)

    assert len(result) == 1
    candidate = result[0]
    assert candidate.state == ["closed"]
    assert candidate.size == ["medium"]
    assert candidate.is_manipulable is True
    assert candidate.is_stateful is True
    assert candidate.exceeds_weight_limit is False


def test_ambiguous_boolean_fields_are_conservative():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="cabinet", is_manipulable=True, is_stateful=True),
            make_object(id="o2", object_class="cabinet", is_manipulable=False, is_stateful=True),
        ]
    )

    result = check_ambiguous(scene)

    assert len(result) == 1
    candidate = result[0]
    assert candidate.is_manipulable is False
    assert candidate.is_stateful is True
    assert candidate.exceeds_weight_limit is False


def test_false_premise_off_monitor():
    scene = build_scene(
        objects=[
            make_object(
                object_class="monitor",
                is_stateful=True,
                state="off",
            )
        ]
    )

    result = check_false_premise(scene)

    assert len(result) == 1
    assert result[0].object_class == "monitor"
    assert result[0].current_state == "off"


def test_false_premise_open_door():
    scene = build_scene(
        objects=[make_object(object_class="door", is_stateful=True, state="open")]
    )

    result = check_false_premise(scene)

    assert len(result) == 1
    assert result[0].current_state == "open"


def test_false_premise_unknown_state():
    scene = build_scene(
        objects=[make_object(is_stateful=True, state="unknown")]
    )

    result = check_false_premise(scene)

    assert result == []


def test_false_premise_null_state():
    scene = build_scene(
        objects=[make_object(is_stateful=True, state=None)]
    )

    result = check_false_premise(scene)

    assert result == []


def test_false_premise_not_stateful():
    scene = build_scene(
        objects=[make_object(is_stateful=False, state="off")]
    )

    result = check_false_premise(scene)

    assert result == []


def test_underspecified_objects_include_manipulable_and_stateful():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="mug", is_manipulable=True, is_stateful=False),
            make_object(id="o2", object_class="door", is_manipulable=False, is_stateful=True, state="open"),
            make_object(id="o3", object_class="wall", is_manipulable=False, is_stateful=False),
        ]
    )

    result = check_underspecified_objects(scene)

    assert [candidate.object_id for candidate in result] == ["o1", "o2"]
    assert result[0].object_class == "mug"
    assert result[1].object_class == "door"
    assert result[1].state == "open"


def test_underspecified_locations_include_all_scene_locations():
    scene = build_scene(
        locations=[
            make_location(id="l1", description="table", location_type="surface", size="medium"),
            make_location(id="l2", description="floor area", location_type="floor_region", size="large"),
        ]
    )

    result = check_underspecified_locations(scene)

    assert [candidate.location_id for candidate in result] == ["l1", "l2"]
    assert result[0].description == "table"
    assert result[1].location_type == "floor_region"


def test_false_premise_state_not_in_map():
    scene = build_scene(
        objects=[make_object(is_stateful=True, state="tilted")]
    )

    result = check_false_premise(scene)

    assert result == []


def test_false_premise_multiple_stateful_objects():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="monitor", is_stateful=True, state="off"),
            make_object(id="o2", object_class="door", is_stateful=True, state="closed"),
        ]
    )

    result = check_false_premise(scene)

    assert len(result) == 2
    assert sorted(candidate.current_state for candidate in result) == ["closed", "off"]


def test_infeasible_object_larger_than_container():
    scene = build_scene(
        objects=[make_object(object_class="keyboard", size="medium", is_manipulable=True)],
        locations=[
            make_location(
                location_type="inside_container",
                size="small",
            )
        ],
    )

    result = check_physically_infeasible(scene)

    assert len(result) == 1
    assert result[0].violation == "object_larger_than_container"


def test_infeasible_object_fits_in_container():
    scene = build_scene(
        objects=[make_object(object_class="marker", size="xsmall", is_manipulable=True)],
        locations=[
            make_location(location_type="inside_container", size="small")
        ],
    )

    result = check_physically_infeasible(scene)

    assert result == []


def test_infeasible_non_manipulable_object():
    scene = build_scene(
        objects=[make_object(object_class="desk", size="xlarge", is_manipulable=False)],
        locations=[
            make_location(location_type="inside_container", size="small")
        ],
    )

    result = check_physically_infeasible(scene)

    assert result == []


def test_infeasible_surface_location_skipped():
    scene = build_scene(
        objects=[make_object(object_class="keyboard", size="medium", is_manipulable=True)],
        locations=[make_location(location_type="surface", size="xlarge")],
    )

    result = check_physically_infeasible(scene)

    assert result == []


def test_infeasible_all_container_types():
    scene = build_scene(
        objects=[make_object(object_class="keyboard", size="medium", is_manipulable=True)],
        locations=[
            make_location(id="l1", location_type="container", size="small"),
            make_location(id="l2", location_type="drawer", size="small"),
            make_location(id="l3", location_type="inside_container", size="small"),
        ],
    )

    result = check_physically_infeasible(scene)

    assert len(result) == 3
    assert all(candidate.violation == "object_larger_than_container" for candidate in result)


def test_infeasible_ignores_weight_limit_violation():
    scene = build_scene(
        objects=[
            make_object(
                object_class="refrigerator",
                size="large",
                location_id="l1",
                exceeds_weight_limit=True,
            )
        ],
        locations=[
            make_location(id="l1", description="office floor", size="xlarge"),
            make_location(id="l2", description="table surface", size="medium"),
            make_location(id="l3", description="warehouse floor", size="xlarge"),
        ],
    )

    result = check_physically_infeasible(scene)

    assert result == []


def test_infeasible_excludes_large_objects_that_exceed_weight_limit():
    scene = build_scene(
        objects=[
            make_object(
                object_class="refrigerator",
                size="xlarge",
                is_manipulable=True,
                exceeds_weight_limit=True,
            )
        ],
        locations=[
            make_location(
                id="l1",
                description="wood cabinet front",
                location_type="drawer",
                size="large",
            ),
        ],
    )

    result = check_physically_infeasible(scene)

    assert result == []


def test_infeasible_weight_limit_only_scene_returns_empty():
    scene = build_scene(
        objects=[
            make_object(
                object_class="refrigerator",
                size="large",
                location_id="l1",
                exceeds_weight_limit=True,
            )
        ],
        locations=[make_location(id="l1", size="xlarge")],
    )

    result = check_physically_infeasible(scene)

    assert result == []


def test_infeasible_weight_and_size_same_scene():
    scene = build_scene(
        objects=[
            make_object(
                id="o1",
                object_class="keyboard",
                size="medium",
                is_manipulable=True,
                exceeds_weight_limit=False,
                location_id="l1",
            ),
            make_object(
                id="o2",
                object_class="refrigerator",
                size="large",
                is_manipulable=False,
                exceeds_weight_limit=True,
                location_id="l1",
            ),
        ],
        locations=[
            make_location(id="l1", description="floor", location_type="surface", size="xlarge"),
            make_location(
                id="l2",
                description="bowl",
                location_type="inside_container",
                size="small",
            ),
        ],
    )

    result = check_physically_infeasible(scene)

    assert len(result) == 1
    assert result[0].object_id == "o1"
    assert result[0].violation == "object_larger_than_container"


def test_missing_capability_olfaction():
    scene = build_scene(objects=[make_object(modalities=["olfaction"])])

    result = check_missing_capability(scene)

    assert len(result) == 1
    assert result[0].required_modality == "olfaction"


def test_missing_capability_audition():
    scene = build_scene(objects=[make_object(modalities=["audition"])])

    result = check_missing_capability(scene)

    assert len(result) == 1
    assert result[0].required_modality == "audition"


def test_missing_capability_vision_present():
    scene = build_scene(objects=[make_object(modalities=["vision"])])

    result = check_missing_capability(scene)

    assert result == []


def test_missing_capability_manipulation_present():
    scene = build_scene(objects=[make_object(modalities=["manipulation"])])

    result = check_missing_capability(scene)

    assert result == []


def test_missing_capability_empty():
    scene = build_scene(objects=[make_object(modalities=[])])

    result = check_missing_capability(scene)

    assert result == []


def test_missing_capability_multiple_modalities():
    scene = build_scene(objects=[make_object(modalities=["olfaction", "audition"])])

    result = check_missing_capability(scene)

    assert len(result) == 2
    assert sorted(candidate.required_modality for candidate in result) == [
        "audition",
        "olfaction",
    ]


def test_missing_capability_multiple_objects():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="stove", modalities=["olfaction"]),
            make_object(id="o2", object_class="speaker", modalities=["audition"]),
        ]
    )

    result = check_missing_capability(scene)

    assert len(result) == 2
    assert sorted(candidate.object_class for candidate in result) == ["speaker", "stove"]


def test_subjective_two_instances():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="chair"),
            make_object(id="o2", object_class="chair"),
        ]
    )

    result = check_subjective(scene)

    assert len(result) == 1
    assert result[0].object_class == "chair"
    assert result[0].instance_ids == ["o1", "o2"]


def test_subjective_three_instances():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="book"),
            make_object(id="o2", object_class="book"),
            make_object(id="o3", object_class="book"),
        ]
    )

    result = check_subjective(scene)

    assert len(result) == 1
    assert result[0].instance_ids == ["o1", "o2", "o3"]


def test_subjective_single_instance():
    scene = build_scene(objects=[make_object(object_class="chair")])

    result = check_subjective(scene)

    assert result == []


def test_subjective_different_classes():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="chair"),
            make_object(id="o2", object_class="table"),
        ]
    )

    result = check_subjective(scene)

    assert result == []


def test_subjective_multiple_classes():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="chair"),
            make_object(id="o2", object_class="chair"),
            make_object(id="o3", object_class="cup"),
            make_object(id="o4", object_class="cup"),
        ]
    )

    result = check_subjective(scene)

    assert len(result) == 2
    assert sorted(candidate.object_class for candidate in result) == ["chair", "cup"]


def test_subjective_independent_of_attributes():
    scene = build_scene(
        objects=[
            make_object(id="o1", object_class="chair", color="black"),
            make_object(id="o2", object_class="chair", color="brown"),
        ]
    )

    result = check_subjective(scene)

    assert len(result) == 1
    assert result[0].object_class == "chair"
