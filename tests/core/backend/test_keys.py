"""多 key 池测试：最少在途调度、失效剔除、指纹不外泄。"""

from __future__ import annotations

import threading

import pytest

from jc.backend.keys import KeyPool, fingerprint
from jc.backend.types import AuthError


def test_fingerprint_is_stable_and_short() -> None:
    first = fingerprint("nvapi-secret")
    assert first == fingerprint("nvapi-secret")
    assert len(first) == 8
    assert first != fingerprint("nvapi-other")


def test_pool_rejects_empty_and_bad_limit() -> None:
    with pytest.raises(ValueError, match="key 池不能为空"):
        KeyPool([])
    with pytest.raises(ValueError, match="max_in_flight_per_key"):
        KeyPool(["a"], 0)


def test_slot_hands_out_keys_and_returns_them() -> None:
    pool = KeyPool(["k1", "k2"], max_in_flight_per_key=1)
    with pool.slot() as key:
        assert key in {"k1", "k2"}
    assert pool.available() == 2


def test_least_in_flight_prefers_idle_key() -> None:
    """嵌套占两个名额时，第二次必须拿到另一个 key（在途少的优先）。"""
    pool = KeyPool(["k1", "k2"], max_in_flight_per_key=1)
    with pool.slot() as first:
        with pool.slot() as second:
            assert first != second


def test_mark_invalid_removes_key_from_rotation() -> None:
    pool = KeyPool(["bad", "good"], max_in_flight_per_key=1)
    pool.mark_invalid("bad")
    assert pool.available() == 1
    for _ in range(3):
        with pool.slot() as key:
            assert key == "good"


def test_all_keys_invalid_raises_auth_error() -> None:
    pool = KeyPool(["only"], max_in_flight_per_key=1)
    pool.mark_invalid("only")
    with pytest.raises(AuthError, match="key 池已空"):
        with pool.slot():
            pass


def test_mark_invalid_is_idempotent() -> None:
    pool = KeyPool(["a", "b"], max_in_flight_per_key=1)
    pool.mark_invalid("a")
    pool.mark_invalid("a")
    assert pool.available() == 1


def test_snapshot_never_leaks_plaintext() -> None:
    pool = KeyPool(["super-secret"], max_in_flight_per_key=1)
    with pool.slot():
        snapshot = pool.snapshot()
    assert list(snapshot) == [fingerprint("super-secret")]
    assert "super-secret" not in str(snapshot)


def test_size_counts_every_key_including_retired_ones() -> None:
    """`size` 是池子原本有几个 key；废掉的也还算在里面，方便算失效比例。"""
    pool = KeyPool(["a", "b"], max_in_flight_per_key=1)
    pool.mark_invalid("a")
    assert pool.size == 2
    assert pool.available() == 1


def test_slot_blocks_until_a_slot_frees() -> None:
    """唯一 key 被占满时，第二个调用方必须**等**，而不是拿到同一个 key 硬闯。"""
    pool = KeyPool(["only"], max_in_flight_per_key=1)
    acquired: list[str] = []
    release = threading.Event()
    first_inside = threading.Event()

    def holder() -> None:
        """输入：无。输出：无。预期行为：占住名额直到 release 被 set。"""
        with pool.slot() as key:
            first_inside.set()
            release.wait(5)
            acquired.append(f"holder:{key}")

    def waiter() -> None:
        """输入：无。输出：无。预期行为：阻塞到 holder 放手后才拿到 key。"""
        with pool.slot() as key:
            acquired.append(f"waiter:{key}")

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert first_inside.wait(5)
    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()
    waiter_thread.join(0.2)
    assert waiter_thread.is_alive()  # 还卡在 slot() 里，说明真的在等
    assert acquired == []
    release.set()
    waiter_thread.join(5)
    holder_thread.join(5)
    assert sorted(acquired) == ["holder:only", "waiter:only"]
