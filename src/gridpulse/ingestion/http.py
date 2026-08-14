from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
RATE_LIMIT_STATUS = 429

MAX_ATTEMPTS = 6
BASE_BACKOFF = 1.5

RATE_LIMIT_BASE_WAIT = 20.0
RATE_LIMIT_MAX_WAIT = 90.0


class UpstreamError(RuntimeError):
    pass


def _retry_delay(status: int | None, attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), RATE_LIMIT_MAX_WAIT)
        except (TypeError, ValueError):
            pass

    if status == RATE_LIMIT_STATUS:
        return min(RATE_LIMIT_BASE_WAIT * attempt, RATE_LIMIT_MAX_WAIT) + random.uniform(0, 3)

    return BASE_BACKOFF**attempt + random.uniform(0, 1)


async def fetch_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | list[tuple[str, Any]],
    *,
    semaphore: asyncio.Semaphore | None = None,
    label: str = "",
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if semaphore is not None:
                async with semaphore:
                    response = await client.get(url, params=params)
            else:
                response = await client.get(url, params=params)

            if response.status_code in RETRY_STATUS:
                if attempt == MAX_ATTEMPTS:
                    raise UpstreamError(
                        f"{label or url} still returning {response.status_code} "
                        f"after {MAX_ATTEMPTS} attempts."
                    )
                delay = _retry_delay(
                    response.status_code, attempt, response.headers.get("Retry-After")
                )
                level = logger.warning if response.status_code == RATE_LIMIT_STATUS else logger.debug
                level(
                    "%s attempt %d/%d got %s%s; waiting %.0fs",
                    label or url, attempt, MAX_ATTEMPTS, response.status_code,
                    " (rate limited)" if response.status_code == RATE_LIMIT_STATUS else "",
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if response.status_code >= 400:
                raise UpstreamError(
                    f"{label or url} returned {response.status_code}: {response.text[:400]}"
                )
            return response.json()

        except UpstreamError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                break
            delay = _retry_delay(None, attempt, None)
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                label or url, attempt, MAX_ATTEMPTS, exc, delay,
            )
            await asyncio.sleep(delay)

    raise UpstreamError(f"{label or url} failed after {MAX_ATTEMPTS} attempts: {last_error}")
