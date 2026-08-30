"""User input queued while a turn is working: the text, its image refs, and the draft.

`Session.pending_user_inputs` holds these; the queue is lock-guarded because input arrives on
one thread while the agent runs on another, and a snapshot round-trips them through `to_json` /
`from_json`.
"""

from __future__ import annotations

from dataclasses import dataclass

from wizolt.base import Json
from wizolt.image import IMAGE_REFS_KEY, ImageRef, UserInput


@dataclass(eq=False)
class QueuedInput:
    text: str
    images: tuple[ImageRef, ...] = ()
    draft: str = ""
    inflight: bool = False

    def to_json(self) -> str | Json:
        if not self.images:
            return self.text
        return {
            "text": self.text,
            "draft": self.draft,
            IMAGE_REFS_KEY: [image.to_json() for image in self.images],
        }

    @classmethod
    def from_json(cls, value: object) -> QueuedInput | None:
        if isinstance(value, str):
            return cls(value) if value.strip() else None
        if not isinstance(value, dict):
            return None
        text = str(value.get("text") or "")
        raw_images = value.get(IMAGE_REFS_KEY)
        images = tuple(image for raw in raw_images if (image := ImageRef.from_json(raw)) is not None) if isinstance(raw_images, list) else ()
        draft = str(value.get("draft") or text)
        if not text.strip():
            return None
        if draft.count("\ufffc") != len(images):
            return cls(text)
        return cls(text, images, draft)

    def user_input(self) -> UserInput:
        return UserInput(self.draft or self.text, self.images)

    def message(self, prefix: str = "") -> Json:
        message: Json = {"role": "user", "content": prefix + self.text}
        if self.images:
            message[IMAGE_REFS_KEY] = [image.to_json() for image in self.images]
        return message
