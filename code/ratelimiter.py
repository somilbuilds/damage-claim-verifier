"""
ratelimiter.py — Per-provider rate limiting for Gemini and Groq APIs.

Gemini: 15 RPM → 4.5s minimum between calls.
Groq:   30 RPM → 2.0s minimum between calls.

On HTTP 429, sleep 60 seconds and retry once.
Sequential only — no thread safety needed.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Minimum seconds between consecutive calls per provider.
_MIN_INTERVAL = {
    "gemini": 4.5,
    "groq": 2.0,
}

# Last call timestamp per provider.
_last_call: dict[str, float] = {
    "gemini": 0.0,
    "groq": 0.0,
}

# Seconds to sleep on a 429 before the single retry.
RATE_LIMIT_BACKOFF = 60


def wait_if_needed(provider: str) -> None:
    """Block until it is safe to call the given provider.

    Args:
        provider: "gemini" or "groq"
    """
    provider = provider.lower()
    interval = _MIN_INTERVAL.get(provider)
    if interval is None:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'gemini' or 'groq'.")

    elapsed = time.monotonic() - _last_call[provider]
    if elapsed < interval:
        sleep_for = interval - elapsed
        logger.debug(
            "Rate limit: sleeping %.2fs before next %s call", sleep_for, provider
        )
        time.sleep(sleep_for)


def record_call(provider: str) -> None:
    """Record that a call was just made to the given provider.

    Call this immediately after the API request returns (success or error).
    """
    provider = provider.lower()
    if provider not in _last_call:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'gemini' or 'groq'.")
    _last_call[provider] = time.monotonic()


def handle_rate_limit(provider: str) -> None:
    """Sleep for the backoff period after receiving a 429.

    The caller should invoke this, then retry the request exactly once.
    """
    provider = provider.lower()
    logger.warning(
        "429 rate limit hit for %s. Sleeping %ds before retry.",
        provider,
        RATE_LIMIT_BACKOFF,
    )
    time.sleep(RATE_LIMIT_BACKOFF)
    # Reset the last-call timer so the retry doesn't double-wait.
    _last_call[provider] = 0.0


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a 429 rate-limit error from either provider.

    Works with both google-genai and groq SDK exception types.
    """
    # Groq raises groq.RateLimitError (subclass with status_code 429).
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True

    # google-genai may raise google.api_core.exceptions.ResourceExhausted
    # or a generic error with "429" in the message.
    exc_name = type(exc).__name__
    if exc_name in ("ResourceExhausted", "TooManyRequests", "RateLimitError"):
        return True

    if "429" in str(exc):
        return True

    return False


def call_with_rate_limit(
    provider: str,
    api_fn,
    *args,
    claim_id: str = "unknown",
    **kwargs,
):
    """Call an API function with rate limiting, 429 retry, and guaranteed no-raise.

    1. Waits until it is safe to call the provider.
    2. Calls api_fn(*args, **kwargs).
    3. On 429: sleeps 60s, retries once.
    4. On second failure (any kind): returns None.
    5. On non-429 error on first attempt: returns None.

    Never raises. Returns the API response on success, or None on failure.
    The caller must check for None and apply safe defaults.

    Args:
        provider: "gemini" or "groq"
        api_fn: Callable that performs the actual API request.
        *args: Positional arguments forwarded to api_fn.
        claim_id: Row identifier for log messages.
        **kwargs: Keyword arguments forwarded to api_fn.

    Returns:
        The return value of api_fn on success, or None on failure.
    """
    provider = provider.lower()

    # --- First attempt ---
    wait_if_needed(provider)
    try:
        result = api_fn(*args, **kwargs)
        record_call(provider)
        return result
    except Exception as first_exc:
        record_call(provider)

        if not is_rate_limit_error(first_exc):
            # Non-429 error: log and return None immediately.
            logger.warning(
                "[%s] API error (non-429) for claim %s on %s: %s",
                provider,
                claim_id,
                provider,
                first_exc,
            )
            return None

        # --- 429 hit: sleep and retry once ---
        logger.warning(
            "[%s] 429 rate limit for claim %s. Sleeping %ds before retry.",
            provider,
            claim_id,
            RATE_LIMIT_BACKOFF,
        )
        handle_rate_limit(provider)

    # --- Second attempt (retry) ---
    wait_if_needed(provider)
    try:
        result = api_fn(*args, **kwargs)
        record_call(provider)
        return result
    except Exception as retry_exc:
        record_call(provider)
        if is_rate_limit_error(retry_exc):
            logger.warning(
                "[%s] REPEATED 429 for claim %s. "
                "Giving up after retry — row will use safe defaults.",
                provider,
                claim_id,
            )
        else:
            logger.warning(
                "[%s] API error on retry for claim %s: %s. "
                "Row will use safe defaults.",
                provider,
                claim_id,
                retry_exc,
            )
        return None
