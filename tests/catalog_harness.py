"""Test helpers for the catalog-backed provider policy.

The resolve/reasoning_choices/normalized_reasoning entry points moved from ProviderConfig onto
the compiled ProviderPolicy held by a session's catalog (with bundled_policy as the offline
fallback). These thin wrappers keep the policy-focused tests readable.
"""

from minacode.providers.compat import bundled_policy


def resolve(config):
    return bundled_policy().resolve(config)


def reasoning_choices(config, model=""):
    return bundled_policy().reasoning_choices(config, model)


def normalized_reasoning(config, model=""):
    return bundled_policy().normalized_reasoning(config, model)
