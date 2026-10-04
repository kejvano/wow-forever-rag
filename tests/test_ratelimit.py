from datetime import date

from ratelimit import DailyCap, RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_the_limit_then_blocks():
    limiter = RateLimiter([(60, 3)], clock=FakeClock())
    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]


def test_window_slides():
    clock = FakeClock()
    limiter = RateLimiter([(60, 2)], clock=clock)
    limiter.allow("a")
    limiter.allow("a")
    assert not limiter.allow("a")
    clock.now = 61
    assert limiter.allow("a")


def test_visitors_are_limited_separately():
    limiter = RateLimiter([(60, 1)], clock=FakeClock())
    assert limiter.allow("a")
    assert limiter.allow("b")
    assert not limiter.allow("a")


def test_every_window_is_enforced():
    clock = FakeClock()
    limiter = RateLimiter([(60, 5), (86400, 6)], clock=clock)
    assert all(limiter.allow("a") for _ in range(5))
    assert not limiter.allow("a")  # per-minute limit
    clock.now = 120
    assert limiter.allow("a")
    assert not limiter.allow("a")  # per-day limit
    clock.now = 86400 + 1
    assert limiter.allow("a")


def test_blocked_requests_do_not_extend_the_block():
    clock = FakeClock()
    limiter = RateLimiter([(60, 1)], clock=clock)
    limiter.allow("a")
    clock.now = 30
    assert not limiter.allow("a")
    clock.now = 60
    assert limiter.allow("a")


def test_daily_cap_blocks_after_the_limit():
    cap = DailyCap(2, today=lambda: date(2026, 11, 4))
    assert [cap.allow() for _ in range(3)] == [True, True, False]


def test_daily_cap_resets_on_a_new_day():
    days = iter([date(2026, 11, 4)] * 3 + [date(2026, 11, 5)])
    cap = DailyCap(2, today=lambda: next(days))
    assert [cap.allow() for _ in range(4)] == [True, True, False, True]
