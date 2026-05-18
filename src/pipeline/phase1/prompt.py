from __future__ import annotations

from vocabs import (
    ATTRIBUTE_VOCAB,
    LOCATION_TYPE_VOCAB,
    MODALITY_VOCAB,
    SIZE_VOCAB,
    STATE_VOCAB,
)


def _format_vocab_block(title: str, values: list[str]) -> str:
    joined = ", ".join(values)
    return f"{title}: {joined}"


def build_prompt() -> str:
    attribute_lines = [
        _format_vocab_block(name.upper(), sorted(values))
        for name, values in ATTRIBUTE_VOCAB.items()
    ]
    state_line = _format_vocab_block("STATE_VOCAB", sorted(STATE_VOCAB))
    size_line = _format_vocab_block("SIZE_VOCAB", SIZE_VOCAB)
    location_line = _format_vocab_block(
        "LOCATION_TYPE_VOCAB", sorted(LOCATION_TYPE_VOCAB)
    )
    modality_line = _format_vocab_block("MODALITY_VOCAB", sorted(MODALITY_VOCAB))

    vocab_section = "\n".join(
        attribute_lines + [state_line, size_line, location_line, modality_line]
    )

    return f"""
You are performing visual grounding for a robotics benchmark.

Return exactly one JSON object and nothing else.
Do not include markdown fences.
Do not include explanations, notes, or trailing text.

Use ONLY the controlled vocabularies below for all vocab-controlled fields.
If a vocab-controlled value cannot be determined from direct visual evidence, use null where allowed.

Controlled vocabularies:
{vocab_section}

Output JSON schema:
{{
  "scene_type": string,
  "scene_objects": [
    {{
      "id": string,         // format: "o<integer>", e.g. "o1"
      "object_class": string,
      "attributes": {{
        "color": string | null,
        "material": string | null,
        "shape": string | null,
        "texture": string | null,
        "pattern": string | null,
        "condition": string | null,
        "style": string | null
      }},
      "state": string | null,
      "size": string,
      "is_manipulable": boolean,
      "is_stateful": boolean,
      "exceeds_weight_limit": boolean,
      "modalities": [string],
      "location_id": string
    }}
  ],
  "scene_locations": [
    {{
      "id": string,         // format: "l<integer>", e.g. "l1"
      "description": string, // short noun phrase, 2-4 words, no leading preposition
      "location_type": string,
      "size": string,
      "contains_object_ids": [string]
    }}
  ],
  "absent_and_implausible_objects": [
    {{
      "object_class": string,
      "color": string | null,
      "state": string | null,
      "size": string,
      "is_manipulable": boolean,
      "is_stateful": boolean,
      "exceeds_weight_limit": boolean
    }}
  ]
}}

Annotation rules:
- Do not include the robot itself or any visible robot body parts
  (for example robot arms, grippers, end-effectors, wheels, or base)
  in scene_objects.

- object_class: most specific common noun a non-expert recognises.
  No brand names. Use underscore for multi-word classes.

- attributes: assign only visually evident values from the corresponding
  ATTRIBUTE vocabularies. Use null for any attribute that is not directly
  visible. Do not infer pattern, condition, or style from object class alone.

- state: from STATE_VOCAB based on direct visual evidence only.
  Assign null for objects with no meaningful discrete state.
  Assign "unknown" if state is present but not determinable.
  Never use LOCATION_TYPE_VOCAB values as state. For example,
  "hanging_point" is a scene_locations.location_type value only,
  not an object state.

- size: use real-world physical size. A distant refrigerator is still "large".

- is_manipulable: true if the object is a discrete movable or operable
  object that a robot could reasonably interact with in principle.
  False for structural elements and fixed built-ins such as walls,
  floors, countertops, and attached fixtures.

- is_stateful: true only if the object has a discrete state
  relevant to manipulation (open/closed, on/off, full/empty).

- exceeds_weight_limit: true if the object likely exceeds the robot's
  lifting or carrying limit of approximately 10kg, even if the object
  is otherwise manipulable in principle.

- Every object must have a location_id referencing a valid entry
    in scene_locations.

- scene_locations.description: use a short noun phrase that can be inserted
  directly into commands such as "put the mug on the {{location}}" or
  "pick up the pen from the {{location}}".
  Do not start with prepositions such as "on", "in", "at", "inside",
  "from", or "near".
  Prefer 2-4 words when possible.
  Good examples: "coffee table", "side table surface", "sofa seat",
  "back wall shelf".
  Bad examples: "on the sofa seating area", "top surface of the coffee table",
  "near the back wall".

- absent_and_implausible_objects: objects that are genuinely absent
  (no instance visible) and implausible for this scene_type.
  Limit to 5 entries.

- For each absent_and_implausible_objects entry:
  - object_class: use the same naming rules as scene_objects.object_class.
  - color: provide one plausible color from COLOR_VOCAB that would make
    a natural language referring expression useful, such as "red mug".
    Use null if no single color is plausibly distinctive.
  - state: provide one plausible manipulation-relevant state from
    STATE_VOCAB for that object class, excluding "unknown".
    Use null if the object is not meaningfully stateful.
  - size: assign the object's typical real-world size from SIZE_VOCAB.
  - is_manipulable: true if a robot could reasonably pick up, hand over,
    or operate the object in principle.
  - is_stateful: true only if the object has a discrete manipulable state.
  - exceeds_weight_limit: true if the object would likely exceed the
    robot's approximate 10kg payload limit.

- Choose absent objects that are clearly not present and that would be
  surprising or implausible in this scene type, so that instructions
  referring to them remain ungroundable even if the model missed some
  visible objects.

- For this benchmark, strongly prefer absent_and_implausible_objects that are
  useful for missing referent instructions:
  - prefer is_manipulable=true
  - prefer exceeds_weight_limit=false
  - prefer size of xsmall, small, or medium
  - avoid large, xlarge, fixed, structural, or obviously immovable objects
    unless no better ordinary object exists
  - prefer ordinary carryable household or everyday objects over bulky items

- Do not include safety-critical, hazardous, violent, illegal, or jailbreak-like
  objects in absent_and_implausible_objects. Exclude examples such as weapons,
  explosives, drugs, restraints, self-harm tools, or other dangerous items.
  This benchmark is not about safety or jailbreak behavior.

- Prefer ordinary household or everyday objects that are simply implausible
  for the given scene and that still support references by color, state, size,
  handover, or lifting constraints when useful.

- modalities: list only the sensing or action modalities
  meaningfully associated with grounding or interacting with that object.
  Every value must come from MODALITY_VOCAB. Infer these from common
  knowledge about the object and the kind of information or interaction
  it affords. Use an empty list if no additional modality is relevant.
  - Use "vision" for objects whose identity, location, visible attributes,
    or visible state can be grounded visually.
  - Use "manipulation" for objects the robot could pick up, move, hand over,
    open, close, turn on, turn off, fill, empty, or otherwise physically operate.
  - Use "olfaction" when smell would be relevant for grounding or interaction,
    such as food, drinks, trash, towels, cleaning items, cups, bottles, or objects
    that may be fresh, stale, sour, dirty, or smelly.
  - Use "audition" for objects that can emit, indicate, or be identified by
    sound, such as timers, phones, alarms, speakers, appliances, buzzers,
    ringing devices, beeping devices, or humming machines.
  - Use "proprioception" when force, resistance, stiffness, weight, or effort
    would be relevant, such as drawers, doors, stuck lids, heavy containers,
    soft objects, firm objects, or objects whose heaviness cannot be known
    from vision alone.
  - Use "thermal_sensing" when temperature would be relevant and cannot be
    reliably determined from vision alone, such as stoves, ovens, pans, kettles,
    toasters, heaters, radiators, hot plates, recently used appliances, hot food,
    or hot drinks.
""".strip()
