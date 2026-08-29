"""Provider request retry policy: error classification, wait computation, and Retry-After parsing."""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import functools
import importlib
import random
import re
from datetime import UTC, datetime
from typing import Any

from minacode.base import (
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    ModelOutputTruncated,
    ModelResponseTimeout,
    ModelStreamIncomplete,
)

_RETRYABLE_STATUS_RE: re.Pattern = re.compile(r"(?:error|status)?[_\s-]*code['\"]?\s*[:=]\s*['\"]?(408|409|425|429|5\d\d)\b")
_STATUS_CODE_RE: re.Pattern = re.compile(r"(?:error|status)?[_\s-]*code['\"]?\s*[:=]\s*['\"]?(4\d\d|5\d\d)\b")

# 429 carries two opposite meanings and cannot be retried uniformly: transient rate limiting
# (retry after backoff is right) and permanent quota/billing failures (retrying just makes the
# user wait through the backoff for an error that will never clear). Compatible endpoints express
# the permanent class as 429 plus account/billing wording in the error body. The marker set below
# deliberately captures shared meanings rather than branching on a provider name or error code.
# This is a fail-open heuristic: unknown wording retries as before, preferring to miss a
# permanent error over misclassifying a transient rate limit as a billing failure.
#
# The markers are phrases, not words, because the vocabulary overlaps: a transient limit is
# commonly phrased with the same nouns the permanent class uses. "Quota exceeded for quota
# metric ... per minute" and similarly named rate-quota errors are per-minute
# limits that clear on their own, so a bare "quota" marker would fail them at once — the exact
# outcome this rule exists to prevent. Same for bare "expired" and "credit", which appear in
# transient infrastructure errors that have nothing to do with an account.
_BILLING_MARKERS: tuple[str, ...] = (
    "insufficient_quota",
    "insufficient balance",
    "insufficient credit",
    "exceeded_current_quota",
    "exceeded your current quota",
    "check your plan",
    "billing",
    "recharge",
    "overdue",
    "arrears",
    "not purchased",
    "notpurchased",
    "allocationquota",
    "account balance",
    "no resource package",
    "package has expired",
    "subscription has expired",
)


def _billing_marker_hit(text: str) -> bool:
    """Fail-open billing-marker scan shared by the SDK-status and text-fallback paths."""
    return any(marker in text for marker in _BILLING_MARKERS)


@functools.cache
def _transport_errors() -> tuple[type[BaseException], ...]:
    """Transport error base classes of every httpx generation present in the environment.

    The provider SDKs moved to httpx2 (openai 3.x, anthropic 1.x) while the MCP client transports
    still speak plain httpx, so both generations run in one process. They are separate exception
    hierarchies — httpx2.TransportError is not a subclass of httpx.TransportError — so matching
    only one silently drops the other's dropped-connection errors out of the retry path. Either
    import may be absent once one side of the migration finishes; the tuple just gets shorter."""
    errors: list[type[BaseException]] = []
    for name in ("httpx", "httpx2"):
        with contextlib.suppress(ImportError):
            errors.append(importlib.import_module(name).TransportError)
    return tuple(errors)


def retryable_error(error: Exception) -> bool:
    """Whether the error is transient enough to retry, minus the deterministic and permanent classes.

    SDK status errors retry on retryable codes, SDK connection/timeout errors and built-in
    network errors always retry, and a fallback parses status codes embedded in the error text or
    cause attributes. A 429 carrying billing wording is a permanent account failure, and a
    truncated generation is deterministic, so both return False. The caller keeps the attempt
    budget and the wait pacing (see retry_delay).
    """
    # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
    import anthropic
    import openai

    # A truncated generation is deterministic: the same request hits the same output cap again.
    if isinstance(error, (ModelResponseTimeout, ModelOutputTruncated)):
        return False
    # A stream that ends without a terminal event is a server-side drop detected after the fact by
    # reassemble_stream, the same class of transient failure as the httpx transport errors below.
    if isinstance(error, ModelStreamIncomplete):
        return True
    cause = getattr(error, "__cause__", None)

    # SDK status errors expose status_code directly. A 429 whose structured error text carries
    # billing wording is a permanent quota/account failure, not a transient limit: fail at once
    # instead of backoff-retrying. Structured text joins code/type/message/body (any may be
    # missing) because the openai SDK unwraps body to the inner error object while the anthropic
    # SDK keeps the full body and only surfaces error.type.
    if isinstance(cause, (openai.APIStatusError, anthropic.APIStatusError)):
        if cause.status_code == 429 and _billing_marker_hit(
            " ".join(str(getattr(cause, field, "") or "") for field in ("code", "type", "message", "body")).lower()
        ):
            return False
        return cause.status_code in {408, 409, 425, 429} or 500 <= cause.status_code < 600

    # SDK connection/timeout errors are always retryable.
    if isinstance(
        cause,
        (openai.APIConnectionError, openai.APITimeoutError, anthropic.APIConnectionError, anthropic.APITimeoutError),
    ):
        return True

    # Built-in network/timeout errors are retryable.
    if isinstance(cause, (TimeoutError, asyncio.TimeoutError, ConnectionError, ConnectionResetError, ConnectionAbortedError)):
        return True

    # Streaming reads surface httpx transport errors unwrapped: the provider SDKs' Stream.__stream__
    # iterates the response directly and re-raises httpx failures (a dropped connection mid-stream is
    # ReadError, an interrupted chunked body is RemoteProtocolError) rather than wrapping them as
    # APIConnectionError. They're the same class of transient failure. httpx transport errors don't
    # inherit OSError, so the isinstance above misses them. See _transport_errors for why this
    # matches both httpx generations rather than the one the SDKs happen to use today.
    if isinstance(cause, _transport_errors()):
        return True

    # Fallback: parse status codes embedded in the error text or cause attributes.
    text = str(error).lower()
    status: Any = getattr(cause, "status_code", None) or getattr(cause, "code", None)
    status_match = _RETRYABLE_STATUS_RE.search(text)
    # The same permanent-429 rule as the SDK branch above, gated on the status the same way:
    # only a 429 can be a billing failure. It runs ahead of the status parses so a "429
    # insufficient balance" message is not rescued by them and retried, but a 5xx that happens
    # to mention an expired certificate stays transient.
    if (str(status) == "429" or (status_match is not None and status_match.group(1) == "429")) and _billing_marker_hit(text):
        return False
    with contextlib.suppress(Exception):
        if int(status) in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
    if status_match:
        return True
    return any(part in text for part in ("internal server error", "timeout", "timed out", "connection reset", "connection aborted", "temporarily unavailable"))


def error_status(error: Exception) -> int | None:
    """The unambiguous HTTP status of a failed request, from the exception/cause chain.

    Structured SDK status wins (ModelClient.call_client wraps SDK errors in ModelError with the
    SDK error as `__cause__`); a rendered status number embedded in the error text is a
    conservative fallback. Returns None when no status can be established, which callers must
    treat as "not eligible" rather than as a status.
    """

    cause = getattr(error, "__cause__", None)
    status: Any = getattr(cause, "status_code", None) or getattr(cause, "code", None)
    with contextlib.suppress(Exception):
        code = int(status)
        if 400 <= code <= 599:
            return code
    match = _STATUS_CODE_RE.search(str(error))
    if match:
        with contextlib.suppress(ValueError):
            return int(match.group(1))
    return None


def retry_reason(error: Exception) -> str:
    """Short, safe label for the retry (status code, or a fixed word) shown in the status bar.

    Never echoes the provider error text: the status bar renders this verbatim, so the label is
    drawn only from the structured status, the embedded status code, or fixed words for the
    timeout/connection/server-error families, with "transient error" as the fallback.
    """
    if isinstance(error, ModelStreamIncomplete):
        return "stream"
    cause = getattr(error, "__cause__", None)
    status: Any = getattr(cause, "status_code", None) or getattr(cause, "code", None)
    with contextlib.suppress(Exception):
        status_code = int(status)
        if 400 <= status_code <= 599:
            return str(status_code)
    text = str(error).lower()
    match = _STATUS_CODE_RE.search(text)
    if match:
        return match.group(1)
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(part in text for part in ("connection", "reset", "aborted")):
        return "connection"
    if "internal server error" in text or "temporarily unavailable" in text:
        return "server error"
    return "transient error"


def retry_after_delay(error: Exception) -> float | None:
    """Provider Retry-After (seconds or HTTP-date) from the SDK cause, clamped to RETRY_MAX_DELAY.

    Returns None to fall back to the backoff algorithm when the header is missing, empty,
    malformed, or negative. Any parsed value is clamped so a single aberrant header cannot stall
    the CLI for minutes; the retry decision itself is unchanged (see retryable_error)."""
    # lazy import: keeps the ~0.8s provider SDK import off the startup path (see the TYPE_CHECKING block above)
    import anthropic
    import openai

    cause = getattr(error, "__cause__", None)
    if not isinstance(cause, (openai.APIStatusError, anthropic.APIStatusError)):
        return None
    headers = getattr(cause.response, "headers", None) or {}
    value = headers.get("retry-after")
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("latin-1")
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = int(text)
    except ValueError:
        # HTTP-date form (RFC 7231; no zone means GMT). parsedate_to_datetime raises on some
        # malformed inputs instead of returning None, so treat either outcome as "unparseable".
        try:
            parsed = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        seconds = (parsed - datetime.now(UTC)).total_seconds()
    if seconds < 0:
        return None
    return min(seconds, RETRY_MAX_DELAY)


def retry_delay(error: Exception, attempt: int) -> float:
    """Single-wait pacing: provider Retry-After wins when parseable, else exponential backoff + jitter."""
    if (retry_after := retry_after_delay(error)) is not None:
        return retry_after
    delay = min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * 2**attempt)
    # jitter 0.5x-1.5x is not optional: without it, parallel read-only tool batches (and worker vs
    # parent requests) would retry in lockstep and spike the provider exactly when it is weakest.
    return delay * (0.5 + random.random())
