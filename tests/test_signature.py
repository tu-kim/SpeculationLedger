"""core.signature 단위 테스트: rolling↔기준 구현 동치, SA 정확성 (D3 지원)."""

import random

from core.signature import (
    MAX_ORDER,
    MIN_ORDER,
    RollingSigStack,
    SuffixAutomaton,
    fold_key,
    sig_of,
)


def test_stack_matches_reference():
    rng = random.Random(1)
    toks = [rng.randrange(1, 1000) for _ in range(64)]
    rs = RollingSigStack()
    for i, t in enumerate(toks):
        rs.push(t)
        for order, sig in rs.stack():
            assert sig == sig_of(toks[i + 1 - order : i + 1])


def test_stack_list_order_convention():
    rs = RollingSigStack()
    rs.push_many([5, 6, 7, 8, 9])
    lst = rs.stack_list()
    assert len(lst) == 5 - MIN_ORDER + 1  # 차수 2..5
    assert lst[0] == sig_of([8, 9])  # index 0 ↔ MIN_ORDER
    assert lst[-1] == sig_of([5, 6, 7, 8, 9])


def test_stack_capped_at_max_order():
    rs = RollingSigStack()
    rs.push_many(list(range(100, 150)))
    assert len(rs.stack_list()) == MAX_ORDER - MIN_ORDER + 1


def test_fold_key_separates_axes():
    sig = sig_of([1, 2, 3])
    keys = {
        fold_key(sig, 3, 111, 0),
        fold_key(sig, 4, 111, 0),
        fold_key(sig, 3, 222, 0),
        fold_key(sig, 3, 111, 1),
    }
    assert len(keys) == 4


def _brute_longest_match(hist: list[int], i: int) -> int:
    """hist[:i]까지 본 상태에서 hist[i] 소비 시 suffix 최장 재발 길이."""
    best = 0
    for ln in range(1, i + 2):
        if ln > i + 1:
            break
        pat = hist[i + 1 - ln : i + 1]
        # 과거(끝이 i-1 이전) 등장 여부
        found = False
        for s in range(0, i + 1 - ln):
            if hist[s : s + ln] == pat:
                found = True
                break
        if found:
            best = ln
        else:
            break
    return best


def test_suffix_automaton_matches_bruteforce():
    rng = random.Random(7)
    toks = [rng.randrange(0, 6) for _ in range(120)]  # 작은 알파벳 → 재발 풍부
    sam = SuffixAutomaton()
    for i, t in enumerate(toks):
        got = sam.extend(t)
        assert got == _brute_longest_match(toks, i), f"pos {i}"
