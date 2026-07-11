"""서명(signature) 구현: rolling hash vs suffix automaton — 결정 D3 (CLAUDE.md §11).

두 구현을 모두 제공하고 sim/bench_d3.py의 μbench로 채택을 결정한다.
채택 결과는 docs/DECISIONS.md D3에 기록.

- RollingSigStack: 차수 2..8 suffix n-gram의 u64 서명 스택을 O(orders)/token으로 유지.
  store 키 = fold_key(sig, order, scope_id, seg) (§3.1 "key u64(suffix hash ⊕ scope ⊕ seg)").
- SuffixAutomaton: 세션 스트림 전체에 대한 online SA — 최장 재발 suffix 길이와
  등장 횟수를 정확히 준다. 서명 스택 대비 정확하지만 메모리/시간 비용이 크다.
"""

from __future__ import annotations

from core.types import U64_MASK

MIN_ORDER = 2
MAX_ORDER = 8

# 홀수 곱수(가역) — splitmix64/fxhash 계열 상수. PYTHONHASHSEED 비의존(I4).
_MULT = 0x9E3779B97F4A7C15
_ORDER_SALT = tuple((0xA076_1D64_78BD_642F * (o + 1)) & U64_MASK for o in range(MAX_ORDER + 1))
_SEG_SALT = tuple((0xE703_7ED1_A0B4_28DB * (s + 3)) & U64_MASK for s in range(4))


def fmix64(x: int) -> int:
    """splitmix64 finalizer — 상위/하위 비트 분산."""
    x &= U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & U64_MASK
    return (x ^ (x >> 31)) & U64_MASK


def sig_of(tokens: list[int] | tuple[int, ...]) -> int:
    """토큰열의 위치-민감 u64 서명 (비증분 기준 구현)."""
    h = 0
    for t in tokens:
        h = (h * _MULT + fmix64(t + 1)) & U64_MASK
    return h


def fold_key(sig: int, order: int, scope_id: int, seg: int) -> int:
    """HotEntry 키: suffix hash ⊕ scope ⊕ seg (+order salt) → fmix64 (§3.1)."""
    return fmix64(sig ^ _ORDER_SALT[order] ^ scope_id ^ _SEG_SALT[seg])


class RollingSigStack:
    """마지막 MAX_ORDER 토큰의 ring buffer에서 차수 2..8 서명 스택을 유지.

    push(tok) 후 stack()은 [(order, sig)]를 고차수→저차수로 반환한다.
    구현은 ring 재계산(차수≤8이라 토큰당 ≤ 8+7+...+2 = 35 mult-add)이다.
    네이티브 구현에서는 진짜 rolling(감산+곱셈)으로 대체 — 수치는 sig_of와 동일해야 하며
    tests/test_signature.py가 동치성을 고정한다.
    """

    __slots__ = ("_ring", "_n")

    def __init__(self) -> None:
        self._ring: list[int] = [0] * MAX_ORDER
        self._n = 0

    def push(self, tok: int) -> None:
        self._ring[self._n % MAX_ORDER] = tok
        self._n += 1

    def push_many(self, toks: list[int] | tuple[int, ...]) -> None:
        for t in toks:
            self.push(t)

    def _tail(self, order: int) -> list[int]:
        n = self._n
        return [self._ring[(n - order + i) % MAX_ORDER] for i in range(order)]

    def stack(self) -> list[tuple[int, int]]:
        """[(order, sig)] — 고차수 우선. 스트림 길이보다 큰 차수는 제외."""
        out = []
        hi = min(self._n, MAX_ORDER)
        for order in range(hi, MIN_ORDER - 1, -1):
            out.append((order, sig_of(self._tail(order))))
        return out

    def stack_list(self) -> list[int]:
        """계약 §3.1 lookup의 sig_stack: list[u64]. index i ↔ 차수 MIN_ORDER+i
        (저차수 우선, 스트림이 짧으면 가능한 차수까지만)."""
        hi = min(self._n, MAX_ORDER)
        return [sig_of(self._tail(order)) for order in range(MIN_ORDER, hi + 1)]

    def clone(self) -> "RollingSigStack":
        c = RollingSigStack()
        c._ring = list(self._ring)
        c._n = self._n
        return c

    def __len__(self) -> int:
        return self._n


class SuffixAutomaton:
    """온라인 suffix automaton (D3 비교 대상 구현).

    extend(tok)로 스트림을 소비하면서, 현재 suffix가 과거에 등장한 최장 길이
    (match_len)와 해당 상태를 추적한다. count는 lazy: finalize_counts() 호출 시
    각 상태의 endpos 크기(등장 횟수)를 링크 트리로 집계한다.
    """

    __slots__ = ("next", "link", "length", "cnt", "last", "_match_state", "_match_len")

    def __init__(self) -> None:
        self.next: list[dict[int, int]] = [{}]
        self.link: list[int] = [-1]
        self.length: list[int] = [0]
        self.cnt: list[int] = [0]
        self.last = 0
        self._match_state = 0
        self._match_len = 0

    def _add_state(self, length: int, link: int, trans: dict[int, int], cnt: int) -> int:
        self.next.append(trans)
        self.link.append(link)
        self.length.append(length)
        self.cnt.append(cnt)
        return len(self.next) - 1

    def extend(self, tok: int) -> int:
        """토큰 1개 소비. 반환값: 소비 *직전* 컨텍스트 기준, tok로 이어지는
        suffix가 과거에 등장했던 최장 match 길이 (재발 측정용)."""
        # --- 재발 질의 (SAM matching 표준 워크) ---
        s, ln = self._match_state, self._match_len
        while s != -1 and tok not in self.next[s]:
            s = self.link[s]
            ln = self.length[s] if s != -1 else 0
        if s == -1:
            self._match_state, self._match_len = 0, 0
            matched = 0
        else:
            self._match_state = self.next[s][tok]
            self._match_len = ln + 1
            matched = self._match_len

        # --- SAM extend 표준 알고리즘 ---
        cur = self._add_state(self.length[self.last] + 1, -1, {}, 1)
        p = self.last
        while p != -1 and tok not in self.next[p]:
            self.next[p][tok] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][tok]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = self._add_state(self.length[p] + 1, self.link[q], dict(self.next[q]), 0)
                while p != -1 and self.next[p].get(tok) == q:
                    self.next[p][tok] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        self.last = cur
        return matched

    def finalize_counts(self) -> None:
        order = sorted(range(len(self.length)), key=lambda i: self.length[i], reverse=True)
        for i in order:
            if self.link[i] >= 0:
                self.cnt[self.link[i]] += self.cnt[i]

    def n_states(self) -> int:
        return len(self.length)

    def approx_bytes(self) -> int:
        trans = sum(len(d) for d in self.next)
        # dict 오버헤드를 뺀 논리 크기: 상태당 (len,link,cnt)=12B + 전이당 8B 상당
        return 12 * len(self.length) + 8 * trans
