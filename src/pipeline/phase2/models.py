from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class ObjectAttributes:
    color: Optional[str]
    material: Optional[str]
    shape: Optional[str]
    texture: Optional[str]
    pattern: Optional[str]
    condition: Optional[str]
    style: Optional[str]


@dataclass
class AmbiguousAttributes:
    color: list[str]
    material: list[str]
    shape: list[str]
    texture: list[str]
    pattern: list[str]
    condition: list[str]
    style: list[str]


@dataclass
class SceneObject:
    id: str
    object_class: str
    attributes: ObjectAttributes
    state: Optional[str]
    size: str
    is_manipulable: bool
    is_stateful: bool
    exceeds_weight_limit: bool
    location_id: str
    modalities: list[str]


@dataclass
class SceneLocation:
    id: str
    description: str
    location_type: str
    size: str
    contains_object_ids: list[str]


@dataclass
class AbsentImplausibleObject:
    object_class: str
    color: Optional[str]
    state: Optional[str]
    size: str
    is_manipulable: bool
    is_stateful: bool
    exceeds_weight_limit: bool


@dataclass
class Phase1Output:
    scene_type: str
    scene_objects: list[SceneObject]
    scene_locations: list[SceneLocation]
    absent_and_implausible_objects: list[AbsentImplausibleObject]


@dataclass
class AmbiguousClass:
    object_class: str
    instance_ids: list[str]
    count: int
    ambiguous_attributes: AmbiguousAttributes
    state: list[str]
    size: list[str]
    is_manipulable: bool
    is_stateful: bool
    exceeds_weight_limit: bool
    distinguishing_attributes: list[str]


@dataclass
class FalsePremiseCandidate:
    object_id: str
    object_class: str
    current_state: str


@dataclass
class InfeasiblePair:
    object_id: str
    object_class: str
    object_size: str
    location_id: str
    location_description: str
    location_size: str
    violation: str


@dataclass
class MissingCapabilityCandidate:
    object_id: str
    object_class: str
    required_modality: str


@dataclass
class SubjectiveClass:
    object_class: str
    instance_ids: list[str]


@dataclass
class UnderspecifiedObjectCandidate:
    object_id: str
    object_class: str
    state: Optional[str]
    size: str
    is_manipulable: bool
    is_stateful: bool
    exceeds_weight_limit: bool
    location_id: str


@dataclass
class UnderspecifiedLocationCandidate:
    location_id: str
    description: str
    location_type: str
    size: str


@dataclass
class Phase2Checks:
    ambiguous_candidates: list[AmbiguousClass]
    false_premise_candidates: list[FalsePremiseCandidate]
    physically_infeasible_pairs: list[InfeasiblePair]
    missing_capability_candidates: list[MissingCapabilityCandidate]
    subjective_candidates: list[SubjectiveClass]
    underspecified_object_candidates: list[UnderspecifiedObjectCandidate]
    underspecified_location_candidates: list[UnderspecifiedLocationCandidate]


@dataclass
class Phase2Output:
    scene_type: str
    scene_objects: list[SceneObject]
    scene_locations: list[SceneLocation]
    absent_and_implausible_objects: list[AbsentImplausibleObject]
    checks: Phase2Checks


def _parse_attributes(data: dict[str, Any]) -> ObjectAttributes:
    return ObjectAttributes(
        color=data.get("color"),
        material=data.get("material"),
        shape=data.get("shape"),
        texture=data.get("texture"),
        pattern=data.get("pattern"),
        condition=data.get("condition"),
        style=data.get("style"),
    )


def _parse_scene_object(data: dict[str, Any]) -> SceneObject:
    return SceneObject(
        id=data["id"],
        object_class=data["object_class"],
        attributes=_parse_attributes(data["attributes"]),
        state=data.get("state"),
        size=data["size"],
        is_manipulable=data["is_manipulable"],
        is_stateful=data["is_stateful"],
        exceeds_weight_limit=data["exceeds_weight_limit"],
        location_id=data["location_id"],
        modalities=list(data.get("modalities", [])),
    )


def _parse_scene_location(data: dict[str, Any]) -> SceneLocation:
    return SceneLocation(
        id=data["id"],
        description=data["description"],
        location_type=data["location_type"],
        size=data["size"],
        contains_object_ids=list(data["contains_object_ids"]),
    )


def _parse_absent_implausible_object(data: dict[str, Any]) -> AbsentImplausibleObject:
    return AbsentImplausibleObject(
        object_class=data["object_class"],
        color=data.get("color"),
        state=data.get("state"),
        size=data["size"],
        is_manipulable=data["is_manipulable"],
        is_stateful=data["is_stateful"],
        exceeds_weight_limit=data["exceeds_weight_limit"],
    )


def parse_phase1_output(data: dict[str, Any]) -> Phase1Output:
    absent_objects = [
        _parse_absent_implausible_object(item)
        for item in data.get("absent_and_implausible_objects", [])
    ]

    return Phase1Output(
        scene_type=data["scene_type"],
        scene_objects=[_parse_scene_object(item) for item in data["scene_objects"]],
        scene_locations=[_parse_scene_location(item) for item in data["scene_locations"]],
        absent_and_implausible_objects=absent_objects,
    )


def _parse_ambiguous_attributes(data: dict[str, Any]) -> AmbiguousAttributes:
    return AmbiguousAttributes(
        color=list(data.get("color", [])),
        material=list(data.get("material", [])),
        shape=list(data.get("shape", [])),
        texture=list(data.get("texture", [])),
        pattern=list(data.get("pattern", [])),
        condition=list(data.get("condition", [])),
        style=list(data.get("style", [])),
    )


def _parse_phase2_checks(data: dict[str, Any]) -> Phase2Checks:
    return Phase2Checks(
        ambiguous_candidates=[
            AmbiguousClass(
                object_class=item["object_class"],
                instance_ids=list(item["instance_ids"]),
                count=item["count"],
                ambiguous_attributes=_parse_ambiguous_attributes(item["ambiguous_attributes"]),
                state=list(item.get("state", [])),
                size=list(item.get("size", [])),
                is_manipulable=item.get("is_manipulable", False),
                is_stateful=item.get("is_stateful", False),
                exceeds_weight_limit=item.get("exceeds_weight_limit", False),
                distinguishing_attributes=list(item["distinguishing_attributes"]),
            )
            for item in data.get("ambiguous_candidates", [])
        ],
        false_premise_candidates=[
            FalsePremiseCandidate(
                object_id=item["object_id"],
                object_class=item["object_class"],
                current_state=item["current_state"],
            )
            for item in data.get("false_premise_candidates", [])
        ],
        physically_infeasible_pairs=[
            InfeasiblePair(
                object_id=item["object_id"],
                object_class=item["object_class"],
                object_size=item["object_size"],
                location_id=item["location_id"],
                location_description=item["location_description"],
                location_size=item["location_size"],
                violation=item["violation"],
            )
            for item in data.get("physically_infeasible_pairs", [])
        ],
        missing_capability_candidates=[
            MissingCapabilityCandidate(
                object_id=item["object_id"],
                object_class=item["object_class"],
                required_modality=item["required_modality"],
            )
            for item in data.get("missing_capability_candidates", [])
        ],
        subjective_candidates=[
            SubjectiveClass(
                object_class=item["object_class"],
                instance_ids=list(item["instance_ids"]),
            )
            for item in data.get("subjective_candidates", [])
        ],
        underspecified_object_candidates=[
            UnderspecifiedObjectCandidate(
                object_id=item["object_id"],
                object_class=item["object_class"],
                state=item.get("state"),
                size=item["size"],
                is_manipulable=item["is_manipulable"],
                is_stateful=item["is_stateful"],
                exceeds_weight_limit=item["exceeds_weight_limit"],
                location_id=item["location_id"],
            )
            for item in data.get("underspecified_object_candidates", [])
        ],
        underspecified_location_candidates=[
            UnderspecifiedLocationCandidate(
                location_id=item["location_id"],
                description=item["description"],
                location_type=item["location_type"],
                size=item["size"],
            )
            for item in data.get("underspecified_location_candidates", [])
        ],
    )


def parse_phase2_output(data: dict[str, Any]) -> Phase2Output:
    return Phase2Output(
        scene_type=data["scene_type"],
        scene_objects=[_parse_scene_object(item) for item in data["scene_objects"]],
        scene_locations=[_parse_scene_location(item) for item in data["scene_locations"]],
        absent_and_implausible_objects=[
            _parse_absent_implausible_object(item)
            for item in data.get("absent_and_implausible_objects", [])
        ],
        checks=_parse_phase2_checks(data.get("checks", {})),
    )


def to_json_dict(value: Any) -> Any:
    return asdict(value)
