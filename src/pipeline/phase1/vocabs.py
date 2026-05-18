from __future__ import annotations

from dataclasses import dataclass


ATTRIBUTE_VOCAB = {
    "color": {
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "silver",
        "gold",
    },
    "material": {
        "wooden",
        "metallic",
        "plastic",
        "glass",
        "ceramic",
        "fabric",
        "rubber",
        "paper",
        "cardboard",
    },
    "shape": {
        "round",
        "rectangular",
        "cylindrical",
        "spherical",
        "flat",
        "tall",
        "wide",
    },
    "texture": {"smooth", "rough", "shiny", "matte", "transparent"},
    "pattern": {"solid", "striped", "spotted", "checked", "floral", "graphic", "plain"},
    "condition": {"new", "worn", "clean", "dirty", "damaged", "fresh"},
    "style": {"simple", "decorative", "modern", "classic", "colorful", "plain"},
}

STATE_VOCAB = {
    "open",
    "closed",
    "full",
    "empty",
    "upright",
    "on",
    "off",
    "lying_flat",
    "unknown",
}

SIZE_VOCAB = ["xsmall", "small", "medium", "large", "xlarge"]

LOCATION_TYPE_VOCAB = {
    "surface",
    "container",
    "floor_region",
    "wall_region",
    "shelf",
    "drawer",
    "inside_container",
    "hanging_point",
}

MODALITY_VOCAB = {
    "olfaction",
    "audition",
    "proprioception",
    "thermal_sensing",
    "manipulation",
    "vision",
}


@dataclass(frozen=True)
class ControlledVocabs:
    attribute: dict[str, set[str]]
    state: set[str]
    size: list[str]
    location_type: set[str]
    modality: set[str]


CONTROLLED_VOCABS = ControlledVocabs(
    attribute=ATTRIBUTE_VOCAB,
    state=STATE_VOCAB,
    size=SIZE_VOCAB,
    location_type=LOCATION_TYPE_VOCAB,
    modality=MODALITY_VOCAB,
)
