import threading
import time
from collections import deque
from datetime import date, datetime, timezone
from typing import Callable


class RateLimiter:
    # limits are (window_seconds, max_requests) pairs, e.g. [(60, 5), (86400, 30)]
    def __init__(self, limits: list[tuple[int, int]], clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.clock = clock
        self.window = max(seconds for seconds, _ in limits)
        self.hits: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self.lock:
            if len(self.hits) > 10_000:
                self._forget_idle(now)
            hits = self.hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            for seconds, max_requests in self.limits:
                if sum(1 for t in hits if now - t < seconds) >= max_requests:
                    return False
            hits.append(now)
            return True

    def _forget_idle(self, now: float) -> None:
        idle = [key for key, hits in self.hits.items() if not hits or now - hits[-1] >= self.window]
        for key in idle:
            del self.hits[key]


class DailyCap:
    def __init__(self, max_per_day: int, today: Callable[[], date] = lambda: datetime.now(timezone.utc).date()):
        self.max_per_day = max_per_day
        self.today = today
        self.day: date | None = None
        self.count = 0
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            day = self.today()
            if day != self.day:
                self.day, self.count = day, 0
            if self.count >= self.max_per_day:
                return False
            self.count += 1
            return True
