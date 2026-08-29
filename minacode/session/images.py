"""The session-local image-delivery decision.

`Session.image_route` answers one question -- is this session's effective main route text-only,
statically or from learned evidence -- so attachments and ViewImage do not duplicate model
matching or 400 learning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minacode.base import IMAGE_ROUTE_TEXT_ONLY_LEARNED, IMAGE_ROUTE_TEXT_ONLY_STATIC, IMAGE_ROUTE_UNKNOWN

if TYPE_CHECKING:
    from minacode.session import Session


class ImageRoute:
    """Unified image-delivery decision for the active main route; session-local.

    Static text-only evidence is folded by the session catalog policy's `resolve()` from the
    compatibility catalog. Learned evidence is created only when an eligible main request
    returns HTTP 400 for a request carrying a current-turn raw image, and is keyed by the full
    route identity (provider entry, resolved API, resolved base URL, model). It lives in memory
    for the live session only: snapshots never carry it and a resumed session starts unknown
    unless the catalog supplies static evidence.

    Attachments and ViewImage must ask this one decision which delivery is required instead of
    duplicating model matching or 400 learning; presentation only observes routing events.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def identity(self) -> tuple[str, str, str, str]:
        """The full identity of the effective active provider entry, used as the learned-evidence key."""

        provider = self.session.config.provider
        resolved = self.session.policy.resolve(provider)
        return (self.session.config.active_provider, resolved.api, resolved.base_url, provider.model.lower())

    def static_text_only(self) -> bool:
        return self.session.policy.resolve(self.session.config.provider).text_only

    def learned_text_only(self) -> bool:
        return self.identity() in self.session.learned_text_only_routes

    def is_text_only(self) -> bool:
        return self.static_text_only() or self.learned_text_only()

    def state(self) -> str:
        if self.static_text_only():
            return IMAGE_ROUTE_TEXT_ONLY_STATIC
        if self.learned_text_only():
            return IMAGE_ROUTE_TEXT_ONLY_LEARNED
        return IMAGE_ROUTE_UNKNOWN

    def learn_text_only(self) -> None:
        """Record session-local evidence for the exact current main route."""

        self.session.learned_text_only_routes.add(self.identity())

    def delivery(self) -> str:
        """How a current image occurrence is delivered: `vision` when the route is text-only
        (static or learned) and a vision entry exists, else a raw attempt on the main model.

        A text-only route without `[vision]` deliberately keeps the raw attempt so the
        provider's real failure stays visible; no local image-disable error is invented.
        """

        if self.is_text_only() and self.session.config.vision_provider:
            return "vision"
        return "raw"
