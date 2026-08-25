"""Vision: explicit image observation through the [vision]-configured provider entry.

A `VisionObserver` sends one non-streaming request to the [vision] entry to describe images for
an explicit ViewImage call or a bridged attachment. Perception only: no tools, no coding task;
the main model does the reasoning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minacode.base import Billing, ModelError
from minacode.image import ImageRef
from minacode.prompts import VISION_OBSERVE_DEFAULT_QUESTION, VISION_OBSERVE_PROMPT

if TYPE_CHECKING:
    from minacode.model.client import ModelClient


class VisionObserver:
    """Ask the [vision]-configured entry to observe images for an explicit ViewImage call.

    Mirrors Compactor.compact(): the [vision] entry is resolved per call and validated locally -- a
    missing field would otherwise surface as a generic SDK credentials error naming nothing the user
    can act on -- then served by one non-streaming api_request with pre-built image blocks.
    """

    def __init__(self, model: ModelClient):
        self.model = model

    def observe(self, images: tuple[ImageRef, ...], question: str = "") -> str:
        self.model.cancel_requested.clear()
        entry_name = self.model.session.config.vision_provider
        provider = self.model.session.config.providers[entry_name]
        if missing := provider.missing_fields():
            raise ModelError(f"vision provider `{entry_name}` is missing {', '.join(missing)}; check [vision] and [provider.{entry_name}]")
        messages = [
            {"role": "system", "content": VISION_OBSERVE_PROMPT},
            {
                "role": "user",
                "content": self.model.session.images.vision_content(images, provider.resolve().api, question.strip() or VISION_OBSERVE_DEFAULT_QUESTION),
            },
        ]
        _, _, content = self.model.api_request(
            messages, tools=None, allow_stream=False, response_timeout=provider.response_timeout, provider=provider, billing=Billing.VISION
        )
        return content.strip()
