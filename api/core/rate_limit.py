from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class RateLimiter:
    """Simple per-user token-bucket using a sliding 60s window.

    Intended for single-process deployments. Not distributed.
    """

    def __init__(self) -> None:
        self.events: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self.window_sec = 60.0

    def _limit_for(self, action: str) -> int:
        if action == "ask":
            return int(os.getenv("RATE_ASK_PER_MIN", "15"))
        if action == "ingest":
            return int(os.getenv("RATE_INGEST_PER_MIN", "6"))
        return int(os.getenv("RATE_DEFAULT_PER_MIN", "60"))

    def allow(self, user_id: str, action: str) -> bool:
        key = (str(user_id), action)
        now = time.time()
        dq = self.events[key]
        # Drop old entries
        while dq and (now - dq[0]) > self.window_sec:
            dq.popleft()
        limit = self._limit_for(action)
        if len(dq) >= limit:
            return False
        dq.append(now)
        return True


rate_limiter = RateLimiter()

