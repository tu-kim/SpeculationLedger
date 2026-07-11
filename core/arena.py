"""V2 span arena + patch + break histogram (CLAUDE.md §3.1 V2).

- SpanArena: append-only 토큰 배열. content-hash dedup, 주기적 compaction(rebuild).
- SpanEntry{arena_off, len} + breaks[{off, patch → HotEntry 키}]:
  과거 verify에서 관측된 break offset 히스토그램과, 그 지점의 correction을 담은
  hot-table 엔트리 키(patch_key)를 기록한다. proposer는 break offset에서 budget을
  pre-split 한다 (§3.2).
- epoch invalidation은 lazy: 읽기 시 비교, 실제 폐기는 compaction에서 (§3.1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _content_hash(tokens: list[int] | tuple[int, ...]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for t in tokens:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.digest()


@dataclass
class Break:
    """span 내부 break 1지점: off 위치에서 기각된 횟수 + correction patch 엔트리 키."""

    count: int = 0
    patch_key: int = 0  # hot-table HotEntry 키 (0 = 미기록)


@dataclass
class SpanEntry:
    """SpanEntry{arena_off u40, len u16} + breaks (§3.1)."""

    key: int  # fold_key ⊕ SPAN_SALT — span 시작 컨텍스트 서명
    arena_off: int
    length: int
    scope_id: int
    seg: int
    dom: int  # epoch domain (file_id, 0=없음)
    epoch: int
    count: int = 1  # 관측(재사용) 횟수 — compaction decay 대상
    breaks: dict[int, Break] = field(default_factory=dict)  # off → Break

    def bytes(self) -> int:
        # 네이티브 레이아웃 추정: key8 + off5 + len2 + epoch2 + hdr1 + count2 + break당 10B
        return 20 + 10 * len(self.breaks)


@dataclass(frozen=True)
class SpanProposal:
    """store.lookup_span 반환형 — proposer가 통째 제안 + break pre-split에 사용."""

    tokens: tuple[int, ...]
    breaks: tuple[tuple[int, int], ...]  # (off, count) 오름차순
    count: int
    key: int


class SpanArena:
    """append-only 토큰 arena. add()는 content-hash로 dedup된 (off, len)을 돌려준다."""

    def __init__(self) -> None:
        self._tokens: list[int] = []
        self._dedup: dict[bytes, int] = {}  # content hash → off

    def add(self, tokens: list[int] | tuple[int, ...]) -> tuple[int, int]:
        h = _content_hash(tokens)
        off = self._dedup.get(h)
        if off is None:
            off = len(self._tokens)
            self._tokens.extend(int(t) for t in tokens)
            self._dedup[h] = off
        return off, len(tokens)

    def get(self, off: int, length: int) -> tuple[int, ...]:
        return tuple(self._tokens[off : off + length])

    def n_tokens(self) -> int:
        return len(self._tokens)

    def bytes(self) -> int:
        return 4 * len(self._tokens)

    def compact(self, live: list[SpanEntry]) -> None:
        """살아있는 span만으로 arena를 재구축하고 SpanEntry.arena_off를 제자리 갱신."""
        fresh = SpanArena()
        for e in live:
            toks = self.get(e.arena_off, e.length)
            e.arena_off, _ = fresh.add(toks)
        self._tokens = fresh._tokens
        self._dedup = fresh._dedup

    def dump(self) -> list[int]:
        return list(self._tokens)

    def load(self, tokens: list[int]) -> None:
        self._tokens = [int(t) for t in tokens]
        self._dedup = {}
        # dedup 맵은 이후 add에서 다시 채워진다 (기존 오프셋 참조는 그대로 유효)
