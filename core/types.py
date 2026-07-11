"""공용 타입: VerifyOutcome, Entry, Scope, Segment (CLAUDE.md §2 core/types.py).

이 모듈은 stdlib만 사용한다. torch/cuda 금지 (I5), numpy도 여기서는 쓰지 않는다 —
online harvester가 GPU에서 넘겨주는 값은 plain int/list로 정규화되어 이 타입으로 들어온다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum

U16_MAX = 0xFFFF
U64_MASK = (1 << 64) - 1


class Segment(IntEnum):
    """seg ∈ {think, tool, code, text} — HotEntry hdr의 2-bit 필드 (§3.1)."""

    THINK = 0
    TOOL = 1
    CODE = 2
    TEXT = 3


class ScopeKind(IntEnum):
    """back-off scope 계층. 값 == scope_depth (session→repo→global, §3.1)."""

    SESSION = 0
    REPO = 1
    GLOBAL = 2  # per-tenant global — tenant 간 공유 금지 (§10 trace 격리와 동일 원칙)


def stable_u64(*parts: str) -> int:
    """플랫폼/실행 무관 결정적 64-bit id. (I4: PYTHONHASHSEED 비의존)"""
    h = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


@dataclass(frozen=True)
class Scope:
    """요청의 scope 좌표 (trace §6 scope 객체와 1:1).

    scope_stack()이 lookup/harvest가 쓰는 (kind, id) 목록을 most-specific-first로 준다.
    모든 tier id에 tenant를 접어 넣어 cross-tenant 조회가 키 공간에서부터 불가능하게 한다(I3).
    """

    tenant: str
    repo: str
    session: str
    instance_id: str = ""

    def session_id(self) -> int:
        return stable_u64("session", self.tenant, self.repo, self.session)

    def repo_id(self) -> int:
        return stable_u64("repo", self.tenant, self.repo)

    def global_id(self) -> int:
        return stable_u64("global", self.tenant)

    def scope_stack(self) -> list[tuple[ScopeKind, int]]:
        return [
            (ScopeKind.SESSION, self.session_id()),
            (ScopeKind.REPO, self.repo_id()),
            (ScopeKind.GLOBAL, self.global_id()),
        ]

    def id_of(self, kind: ScopeKind) -> int:
        if kind is ScopeKind.SESSION:
            return self.session_id()
        if kind is ScopeKind.REPO:
            return self.repo_id()
        return self.global_id()


# --- logp 8-bit 양자화 codec (§3.3 "topk_logp 8-bit 양자화", trace §6 topk_logp_q8) ---
# q = clamp(round(-logp * 16), 0, 255)  →  해상도 1/16 nat, 표현범위 logp ∈ [-15.9375, 0].
_Q8_SCALE = 16.0


def logp_to_q8(logp: float) -> int:
    # 온라인 harvester 경계: -inf(확률 0)·NaN은 최저 신뢰(255), +inf는 0으로 고정
    if logp != logp or logp == float("-inf"):
        return 255
    if logp == float("inf"):
        return 0
    q = round(-logp * _Q8_SCALE)
    return 0 if q < 0 else (255 if q > 255 else int(q))


def q8_to_logp(q: int) -> float:
    return -q / _Q8_SCALE


def q8_to_p(q: int) -> float:
    """q8 → 확률값 (exp). 후보 간 상대비교/정규화 전용."""
    import math

    return math.exp(-q / _Q8_SCALE)


@dataclass(frozen=True)
class VerifyOutcome:
    """verify step 직후 seq 하나의 수확물 (§3.3 계약).

    계약 명시 필드: (draft_ids, accepted_len, bonus_id, topk_ids[pos], topk_logp[pos] q8, seg[pos]).
    positions 축은 draft 위치 0..len(draft_ids)-1 에 correction/bonus 위치 1개를 더한 길이다.
      - p < accepted_len          : draft_ids[p] 수락됨 (realized = draft_ids[p])
      - p == accepted_len < len   : draft_ids[p] 기각, realized = bonus_id (correction)
      - accepted_len == len(draft): 전량 수락, realized 마지막 토큰 = bonus_id (free token)
    topk_*는 realized 경로의 각 position에서 target 분포 top-k (p ∈ 0..accepted_len).

    추가 필드(계약 비명시, docs/DECISIONS.md A-4에 기록):
      - ctx_tail: draft 시작 직전 context 꼬리 토큰들 (signature 계산용, 최소 MAX_ORDER-1개)
      - file_id : seg==CODE 구간이 종속된 파일의 epoch domain (0 = 없음). SegmentFSM이 부여.
    """

    scope: Scope
    ctx_tail: tuple[int, ...]
    draft_ids: tuple[int, ...]
    accepted_len: int
    bonus_id: int
    topk_ids: tuple[tuple[int, ...], ...]
    topk_logp_q8: tuple[tuple[int, ...], ...]
    seg: tuple[int, ...]
    file_id: int = 0
    t_us: int = 0

    def realized(self) -> list[int]:
        """이 verify step이 실제로 커밋한 토큰열 (accepted prefix + bonus/correction)."""
        return list(self.draft_ids[: self.accepted_len]) + [self.bonus_id]


@dataclass(frozen=True)
class PosteriorCand:
    """lookup이 돌려주는 후보 1개.

    p_acc  : 이 위치·이 후보의 blend된 수락(=realized) 확률 추정 (Beta-smoothed acc/(acc+rej))
    p_hat  : blend된 target 분포 확률 p̂ — correction 분포는 별도 필드가 아니라 p̂ 자체다 (§3.1:
             greedy correction = p̂ argmax, T>0 correction 분포 = p̂).
    support: 최강 단일 소스의 관측 수 (acc+rej). 소스 간 합산이 아니다 — 같은 관측이
             (차수×scope)개 소스에 중복 계상되는 것을 막는다.
    """

    tok: int
    p_acc: float
    p_hat: float
    support: int


@dataclass(frozen=True)
class Posterior:
    """단일 (context, scope, seg) 지점의 blend된 사후 분포 (§3.1 lookup 반환형)."""

    cands: tuple[PosteriorCand, ...]  # p_acc·p_hat 결합 점수 내림차순
    weight: float  # blend에 참여한 λ 질량 합 — 신뢰도 지표
    best_order: int  # 매치된 최장 suffix 차수 (진단/proposer 정책용)

    def argmax(self) -> PosteriorCand | None:
        return self.cands[0] if self.cands else None


@dataclass
class LedgerStats:
    """stats() 반환형 (§3.1): hit_rate, entries, bytes, queue_depth, drops."""

    lookups: int = 0
    hits: int = 0
    entries: int = 0
    bytes: int = 0
    queue_depth: int = 0
    drops: int = 0
    harvested_events: int = 0
    stale_skips: int = 0
    compactions: int = 0
    evictions: int = 0
    span_entries: int = 0
    arena_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict:
        d = {
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 6),
            "entries": self.entries,
            "bytes": self.bytes,
            "queue_depth": self.queue_depth,
            "drops": self.drops,
            "harvested_events": self.harvested_events,
            "stale_skips": self.stale_skips,
            "compactions": self.compactions,
            "evictions": self.evictions,
            "span_entries": self.span_entries,
            "arena_tokens": self.arena_tokens,
        }
        return d


@dataclass(frozen=True)
class InvalidationEvent:
    """SegmentFSM이 write/edit tool-call 생성 완료 시점에 방출 (§3.4) → store.bump_epoch."""

    scope: Scope
    file_path: str

    def file_id(self) -> int:
        # u32 domain id — HotEntry.dom 폭과 일치 (0은 "domain 없음" 예약)
        fid = stable_u64("file", self.file_path) & 0xFFFFFFFF
        return fid or 1


@dataclass(frozen=True)
class DraftNode:
    """DraftTree의 노드. parent == -1 이면 현재 컨텍스트 마지막 토큰의 자식(루트 레벨)."""

    tok: int
    parent: int


@dataclass
class DraftTree:
    """propose(ctx, budget) 반환형 (§3.2). flat-array 트리 — vLLM v1 스타일.

    nodes[i].parent < i 를 항상 만족한다(위상 정렬). chain은 폭 1 트리의 특수형.
    """

    nodes: list[DraftNode] = field(default_factory=list)

    def add(self, tok: int, parent: int = -1) -> int:
        assert parent < len(self.nodes)
        self.nodes.append(DraftNode(tok, parent))
        return len(self.nodes) - 1

    def __len__(self) -> int:
        return len(self.nodes)

    def chain(self) -> list[int]:
        """폭 1 트리를 토큰 리스트로 (검증·테스트 편의)."""
        out, parent = [], -1
        for i, n in enumerate(self.nodes):
            if n.parent == parent:
                out.append(n.tok)
                parent = i
        return out
