"""Per-client rate limiting.

The concurrency semaphore bounds how many generations run at once but says
nothing about request *rate*: a single client can still queue thousands of
requests and exhaust a free-tier generation quota. This is a token bucket per
client, held in process memory.

In-process state is the right scope here because the app runs as a single worker
(each worker would otherwise load its own copy of the models). If this ever
scales to multiple workers or replicas, the counter belongs in Redis instead.
"""

from __future__ import annotations

import time
from collections import OrderedDict


class RateLimiter:
    """Token bucket allowing ``burst`` requests, refilling at ``per_minute``."""

    def __init__(self, per_minute: int, burst: int, max_clients: int = 4096) -> None:
        self.rate = per_minute / 60.0
        self.burst = float(burst)
        self.max_clients = max_clients
        # LRU-ordered so the bookkeeping cannot grow without bound under a
        # spray of distinct source addresses.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def check(self, client: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        tokens, last = self._buckets.get(client, (self.burst, now))

        tokens = min(self.burst, tokens + (now - last) * self.rate)

        if tokens < 1.0:
            self._buckets[client] = (tokens, now)
            self._buckets.move_to_end(client)
            return False, max(1.0, (1.0 - tokens) / self.rate)

        self._buckets[client] = (tokens - 1.0, now)
        self._buckets.move_to_end(client)

        while len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)

        return True, 0.0
