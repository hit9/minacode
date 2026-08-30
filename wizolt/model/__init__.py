"""wizolt model layer: the provider client and its wire protocols.

The client owns request lifecycle, retries, and usage accounting; chat.py, responses.py and
anthropic.py build and parse each provider's wire payloads. Higher-layer consumers of the client
(compaction, vision) live in the wizolt package proper.
"""

from wizolt.model.client import ModelClient, PreparedRequest

__all__ = ["ModelClient", "PreparedRequest"]
