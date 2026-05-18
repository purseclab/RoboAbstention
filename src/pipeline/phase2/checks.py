from __future__ import annotations

from collections import Counter, defaultdict

from models import (
    AmbiguousClass,
    AmbiguousAttributes,
    FalsePremiseCandidate,
    InfeasiblePair,
    MissingCapabilityCandidate,
    Phase1Output,
    SubjectiveClass,
    UnderspecifiedLocationCandidate,
    UnderspecifiedObjectCandidate,
)
from size_order import parse_size
from state_action_map import STATE_TO_ACTION


ROBOT_CAPABILITIES: set[str] = {"vision", "manipulation"}
CONTAINER_LOCATION_TYPES = {"container", "drawer", "inside_container"}
ATTRIBUTE_KEYS = (
    "color",
    "material",
    "shape",
    "texture",
    "pattern",
    "condition",
    "style",
)


def _repeated_values(values: list[str | None]) -> list[str]:
    filtered = [value for value in values if value is not None]
    value_counts = Counter(filtered)
    return sorted(value for value, count in value_counts.items() if count > 1)


def _all_share_boolean(values: list[bool]) -> bool:
    return bool(values) and len(set(values)) == 1 and values[0]


def check_ambiguous(scene: Phase1Output) -> list[AmbiguousClass]:
    grouped: dict[str, list] = defaultdict(list)
    for obj in scene.scene_objects:
        grouped[obj.object_class].append(obj)

    results: list[AmbiguousClass] = []
    for object_class, instances in grouped.items():
        if len(instances) <= 1:
            continue

        ambiguous_values: dict[str, list[str]] = {
            attribute_name: [] for attribute_name in ATTRIBUTE_KEYS
        }
        distinguishing_attributes: list[str] = []

        for attribute_name in ATTRIBUTE_KEYS:
            values = [
                getattr(instance.attributes, attribute_name)
                for instance in instances
                if getattr(instance.attributes, attribute_name) is not None
            ]
            repeated_values = _repeated_values(values)
            if repeated_values:
                ambiguous_values[attribute_name] = repeated_values
            non_null_values = set(values)
            if values and len(values) == len(instances) and len(non_null_values) == len(instances):
                distinguishing_attributes.append(attribute_name)

        ambiguous_states = _repeated_values([instance.state for instance in instances])
        ambiguous_sizes = _repeated_values([instance.size for instance in instances])
        all_manipulable = _all_share_boolean(
            [instance.is_manipulable for instance in instances]
        )
        all_stateful = _all_share_boolean(
            [instance.is_stateful for instance in instances]
        )
        all_exceed_weight_limit = _all_share_boolean(
            [instance.exceeds_weight_limit for instance in instances]
        )

        results.append(
            AmbiguousClass(
                object_class=object_class,
                instance_ids=[instance.id for instance in instances],
                count=len(instances),
                ambiguous_attributes=AmbiguousAttributes(**ambiguous_values),
                state=ambiguous_states,
                size=ambiguous_sizes,
                is_manipulable=all_manipulable,
                is_stateful=all_stateful,
                exceeds_weight_limit=all_exceed_weight_limit,
                distinguishing_attributes=distinguishing_attributes,
            )
        )

    return results


def check_false_premise(scene: Phase1Output) -> list[FalsePremiseCandidate]:
    results: list[FalsePremiseCandidate] = []
    for obj in scene.scene_objects:
        if not obj.is_stateful:
            continue
        if obj.state is None or obj.state == "unknown":
            continue
        if obj.state not in STATE_TO_ACTION:
            continue
        results.append(
            FalsePremiseCandidate(
                object_id=obj.id,
                object_class=obj.object_class,
                current_state=obj.state,
            )
        )
    return results


def check_physically_infeasible(scene: Phase1Output) -> list[InfeasiblePair]:
    results: list[InfeasiblePair] = []
    container_locations = [
        location
        for location in scene.scene_locations
        if location.location_type in CONTAINER_LOCATION_TYPES
    ]

    for obj in scene.scene_objects:
        if obj.is_manipulable and not obj.exceeds_weight_limit:
            object_size = parse_size(obj.size)
            for location in container_locations:
                if object_size > parse_size(location.size):
                    results.append(
                        InfeasiblePair(
                            object_id=obj.id,
                            object_class=obj.object_class,
                            object_size=obj.size,
                            location_id=location.id,
                            location_description=location.description,
                            location_size=location.size,
                            violation="object_larger_than_container",
                        )
                    )

    return results


def check_missing_capability(scene: Phase1Output) -> list[MissingCapabilityCandidate]:
    results: list[MissingCapabilityCandidate] = []
    for obj in scene.scene_objects:
        for modality in obj.modalities:
            if modality not in ROBOT_CAPABILITIES:
                results.append(
                    MissingCapabilityCandidate(
                        object_id=obj.id,
                        object_class=obj.object_class,
                        required_modality=modality,
                    )
                )
    return results


def check_subjective(scene: Phase1Output) -> list[SubjectiveClass]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for obj in scene.scene_objects:
        grouped[obj.object_class].append(obj.id)

    results: list[SubjectiveClass] = []
    for object_class, instance_ids in grouped.items():
        if len(instance_ids) <= 1:
            continue
        results.append(
            SubjectiveClass(
                object_class=object_class,
                instance_ids=instance_ids,
            )
        )
    return results


def check_underspecified_objects(scene: Phase1Output) -> list[UnderspecifiedObjectCandidate]:
    results: list[UnderspecifiedObjectCandidate] = []
    for obj in scene.scene_objects:
        if not obj.is_manipulable and not obj.is_stateful:
            continue
        results.append(
            UnderspecifiedObjectCandidate(
                object_id=obj.id,
                object_class=obj.object_class,
                state=obj.state,
                size=obj.size,
                is_manipulable=obj.is_manipulable,
                is_stateful=obj.is_stateful,
                exceeds_weight_limit=obj.exceeds_weight_limit,
                location_id=obj.location_id,
            )
        )
    return results


def check_underspecified_locations(scene: Phase1Output) -> list[UnderspecifiedLocationCandidate]:
    return [
        UnderspecifiedLocationCandidate(
            location_id=location.id,
            description=location.description,
            location_type=location.location_type,
            size=location.size,
        )
        for location in scene.scene_locations
    ]
