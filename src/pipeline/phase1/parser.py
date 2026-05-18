from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from vocabs import (
    ATTRIBUTE_VOCAB,
    LOCATION_TYPE_VOCAB,
    MODALITY_VOCAB,
    SIZE_VOCAB,
    STATE_VOCAB,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class ParseError(Exception):
    message: str
    raw_response: str
    validation_errors: list[str]

    def __str__(self) -> str:
        return self.message


def _strip_markdown_fences(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    if text.endswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _normalize_string(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _normalize_vocab_value(
    value: Any,
    valid_values: set[str],
    field_name: str,
    warnings: list[str],
) -> str | None:
    if value is None:
        return None
    normalized = _normalize_string(value)
    if normalized in valid_values:
        return normalized
    warning = f"{field_name}: invalid value {value!r} normalized to null"
    warnings.append(warning)
    LOGGER.warning(warning)
    return None


def _normalize_state(value: Any, warnings: list[str]) -> str | None:
    if value is None:
        return None
    normalized = _normalize_string(value)
    if normalized in STATE_VOCAB:
        return normalized
    if normalized in LOCATION_TYPE_VOCAB:
        warnings.append(
            f"state: location_type value {value!r} normalized to null"
        )
        return None
    warning = f"state: invalid value {value!r} normalized to 'unknown'"
    warnings.append(warning)
    LOGGER.warning(warning)
    return "unknown"


def _normalize_modalities(value: Any, warnings: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        warnings.append("modalities: non-list value normalized to empty list")
        LOGGER.warning("modalities: non-list value normalized to empty list")
        return []

    normalized_modalities: list[str] = []
    for item in value:
        normalized = _normalize_string(item)
        if normalized in MODALITY_VOCAB:
            if normalized not in normalized_modalities:
                normalized_modalities.append(normalized)
        else:
            warning = f"modalities: invalid value {item!r} dropped"
            warnings.append(warning)
            LOGGER.warning(warning)
    return normalized_modalities


def _require_keys(obj: dict[str, Any], keys: set[str], context: str, errors: list[str]) -> None:
    missing = sorted(keys - set(obj))
    if missing:
        errors.append(f"{context} missing required keys: {missing}")


def _normalize_object(obj: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    local_errors: list[str] = []
    required_keys = {
        "id",
        "object_class",
        "attributes",
        "state",
        "size",
        "is_manipulable",
        "is_stateful",
        "exceeds_weight_limit",
        "modalities",
        "location_id",
    }
    _require_keys(obj, required_keys, "scene_object", local_errors)
    if local_errors:
        errors.extend(local_errors)
        return obj

    attributes = obj.get("attributes")
    if not isinstance(attributes, dict):
        errors.append("scene_object.attributes must be an object")
        return obj

    _require_keys(attributes, set(ATTRIBUTE_VOCAB), "scene_object.attributes", errors)

    normalized_attributes = {}
    for field_name, valid_values in ATTRIBUTE_VOCAB.items():
        normalized_attributes[field_name] = _normalize_vocab_value(
            attributes.get(field_name),
            valid_values,
            f"attributes.{field_name}",
            warnings,
        )

    normalized_size = _normalize_string(obj.get("size"))
    if normalized_size not in SIZE_VOCAB:
        errors.append(f"scene_object.size has invalid value: {obj.get('size')!r}")

    normalized = {
        "id": _normalize_string(obj.get("id")),
        "object_class": _normalize_string(obj.get("object_class")),
        "attributes": normalized_attributes,
        "state": _normalize_state(obj.get("state"), warnings),
        "size": normalized_size,
        "is_manipulable": obj.get("is_manipulable"),
        "is_stateful": obj.get("is_stateful"),
        "exceeds_weight_limit": obj.get("exceeds_weight_limit"),
        "modalities": _normalize_modalities(
            obj.get("modalities"),
            warnings,
        ),
        "location_id": _normalize_string(obj.get("location_id")),
    }

    if not isinstance(normalized["id"], str) or not normalized["id"]:
        errors.append("scene_object.id must be a non-empty string")
    if not isinstance(normalized["object_class"], str) or not normalized["object_class"]:
        errors.append("scene_object.object_class must be a non-empty string")
    if not isinstance(normalized["location_id"], str) or not normalized["location_id"]:
        errors.append("scene_object.location_id must be a non-empty string")
    if not isinstance(normalized["is_manipulable"], bool):
        errors.append("scene_object.is_manipulable must be a boolean")
    if not isinstance(normalized["is_stateful"], bool):
        errors.append("scene_object.is_stateful must be a boolean")
    if not isinstance(normalized["exceeds_weight_limit"], bool):
        errors.append("scene_object.exceeds_weight_limit must be a boolean")

    return normalized


def _normalize_location(
    location: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    local_errors: list[str] = []
    required_keys = {
        "id",
        "description",
        "location_type",
        "size",
        "contains_object_ids",
    }
    _require_keys(location, required_keys, "scene_location", local_errors)
    if local_errors:
        errors.extend(local_errors)
        return location

    normalized_location_type = _normalize_string(location.get("location_type"))
    if normalized_location_type not in LOCATION_TYPE_VOCAB:
        errors.append(
            f"scene_location.location_type has invalid value: {location.get('location_type')!r}"
        )

    normalized_size = _normalize_string(location.get("size"))
    if normalized_size not in SIZE_VOCAB:
        errors.append(f"scene_location.size has invalid value: {location.get('size')!r}")

    contains_ids = location.get("contains_object_ids")
    if not isinstance(contains_ids, list):
        errors.append("scene_location.contains_object_ids must be a list")
        contains_ids = []

    normalized = {
        "id": _normalize_string(location.get("id")),
        "description": _normalize_string(location.get("description")),
        "location_type": normalized_location_type,
        "size": normalized_size,
        "contains_object_ids": [_normalize_string(item) for item in contains_ids],
    }

    if not isinstance(normalized["id"], str) or not normalized["id"]:
        errors.append("scene_location.id must be a non-empty string")
    if not isinstance(normalized["description"], str) or not normalized["description"]:
        errors.append("scene_location.description must be a non-empty string")

    return normalized


def _normalize_absent_object(
    obj: dict[str, Any], errors: list[str], warnings: list[str]
) -> dict[str, Any]:
    local_errors: list[str] = []
    required_keys = {
        "object_class",
        "color",
        "state",
        "size",
        "is_manipulable",
        "is_stateful",
        "exceeds_weight_limit",
    }
    _require_keys(obj, required_keys, "absent_and_implausible_object", local_errors)
    if local_errors:
        errors.extend(local_errors)
        return obj

    normalized_size = _normalize_string(obj.get("size"))
    if normalized_size not in SIZE_VOCAB:
        errors.append(
            "absent_and_implausible_object.size has invalid value: "
            f"{obj.get('size')!r}"
        )

    normalized = {
        "object_class": _normalize_string(obj.get("object_class")),
        "color": _normalize_vocab_value(
            obj.get("color"),
            ATTRIBUTE_VOCAB["color"],
            "absent_and_implausible_object.color",
            warnings,
        ),
        "state": _normalize_vocab_value(
            obj.get("state"),
            STATE_VOCAB - {"unknown"},
            "absent_and_implausible_object.state",
            warnings,
        ),
        "size": normalized_size,
        "is_manipulable": obj.get("is_manipulable"),
        "is_stateful": obj.get("is_stateful"),
        "exceeds_weight_limit": obj.get("exceeds_weight_limit"),
    }

    if not isinstance(normalized["object_class"], str) or not normalized["object_class"]:
        errors.append("absent_and_implausible_object.object_class must be a non-empty string")
    if not isinstance(normalized["is_manipulable"], bool):
        errors.append("absent_and_implausible_object.is_manipulable must be a boolean")
    if not isinstance(normalized["is_stateful"], bool):
        errors.append("absent_and_implausible_object.is_stateful must be a boolean")
    if not isinstance(normalized["exceeds_weight_limit"], bool):
        errors.append("absent_and_implausible_object.exceeds_weight_limit must be a boolean")

    return normalized


def parse_and_validate(raw_response: str) -> dict[str, Any]:
    cleaned = _strip_markdown_fences(raw_response)
    warnings: list[str] = []

    if not cleaned:
        raise ParseError(
            message="VLM response was empty after normalization",
            raw_response=raw_response,
            validation_errors=["The model returned an empty or fence-only response."],
        )

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        details = [str(exc)]
        if raw_response.strip().startswith("```"):
            details.append(
                "The response started with a markdown code fence; the model may have returned fenced or truncated JSON."
            )
        raise ParseError(
            message="Failed to parse VLM response as JSON",
            raw_response=raw_response,
            validation_errors=details,
        ) from exc

    if not isinstance(parsed, dict):
        raise ParseError(
            message="Top-level response must be a JSON object",
            raw_response=raw_response,
            validation_errors=[f"Expected object, got {type(parsed).__name__}"],
        )

    errors: list[str] = []
    required_top_level = {
        "scene_type",
        "scene_objects",
        "scene_locations",
        "absent_and_implausible_objects",
    }
    _require_keys(parsed, required_top_level, "top-level", errors)

    scene_type = _normalize_string(parsed.get("scene_type"))
    scene_objects = parsed.get("scene_objects")
    scene_locations = parsed.get("scene_locations")
    absent_objects = parsed.get("absent_and_implausible_objects")

    if not isinstance(scene_type, str) or not scene_type:
        errors.append("scene_type must be a non-empty string")
    if not isinstance(scene_objects, list):
        errors.append("scene_objects must be a list")
        scene_objects = []
    if not isinstance(scene_locations, list):
        errors.append("scene_locations must be a list")
        scene_locations = []
    if not isinstance(absent_objects, list):
        errors.append("absent_and_implausible_objects must be a list")
        absent_objects = []

    normalized_objects = [
        _normalize_object(obj, errors, warnings)
        for obj in scene_objects
        if isinstance(obj, dict)
    ]
    non_object_items = len(scene_objects) - len(normalized_objects)
    if non_object_items:
        errors.append("scene_objects must contain only objects")

    normalized_locations = [
        _normalize_location(location, errors)
        for location in scene_locations
        if isinstance(location, dict)
    ]
    non_location_items = len(scene_locations) - len(normalized_locations)
    if non_location_items:
        errors.append("scene_locations must contain only objects")

    object_ids = [obj.get("id") for obj in normalized_objects]
    location_ids = [location.get("id") for location in normalized_locations]

    duplicate_object_ids = sorted({item for item in object_ids if object_ids.count(item) > 1})
    duplicate_location_ids = sorted(
        {item for item in location_ids if location_ids.count(item) > 1}
    )
    if duplicate_object_ids:
        errors.append(f"Duplicate object ids: {duplicate_object_ids}")
    if duplicate_location_ids:
        errors.append(f"Duplicate location ids: {duplicate_location_ids}")

    location_id_set = set(location_ids)
    object_id_set = set(object_ids)

    for obj in normalized_objects:
        if obj.get("location_id") not in location_id_set:
            errors.append(
                f"Object {obj.get('id')} references missing location_id {obj.get('location_id')}"
            )

    for location in normalized_locations:
        for object_id in location.get("contains_object_ids", []):
            if object_id not in object_id_set:
                errors.append(
                    f"Location {location.get('id')} references missing object id {object_id}"
                )

    normalized_absent_objects = [
        _normalize_absent_object(obj, errors, warnings)
        for obj in absent_objects
        if isinstance(obj, dict)
    ]
    non_absent_object_items = len(absent_objects) - len(normalized_absent_objects)
    if non_absent_object_items:
        errors.append("absent_and_implausible_objects must contain only objects")

    if errors:
        raise ParseError(
            message="Phase 1 response failed validation",
            raw_response=raw_response,
            validation_errors=errors,
        )

    return {
        "scene_type": scene_type,
        "scene_objects": normalized_objects,
        "scene_locations": normalized_locations,
        "absent_and_implausible_objects": normalized_absent_objects,
    }
