"""LedgerStore: 검증 산출물(verify outcome)의 host-memory 영속 저장소 (CLAUDE.md §3.1).

계약 API:
    lookup(sig_stack: list[u64], scope_stack: list[ScopeId], seg) -> Posterior | None
    harvest(events: list[VerifyOutcome]) -> None      # 절대 블로킹 금지 (bounded queue)
    bump_epoch(scope: ScopeId, file_id: u32) -> None  # O(1)
    snapshot(path) / load(path)
    stats() -> LedgerStats

- HotEntry: key u64(suffix hash ⊕ scope ⊕ seg) · epoch u16 · hdr(k|seg|flags) ·
  cand[k]{tok, logp̂ q8, acc u16, rej u16}. k ∈ 1..16, top-k coverage ≥ 0.9까지 적응 확장.
- 이중 back-off: 차수 2..8 × scope 3계층 → core.backoff.blend가 단일 λ로 통합.
- correction 필드 없음: greedy correction = p̂ argmax, T>0 correction 분포 = p̂ (§3.1).
- V2(version=2): SpanEntry + arena + break 히스토그램 (core.arena).
- epoch invalidation lazy: 읽기 시 비교(stale skip), 실제 폐기는 compaction.
- 이 구현은 sim/online 공용 단일 진실이다. Python 참조 구현이며 G4 μs 계약은
  Phase 1에서 동일 로직의 native 포팅이 진다 (레이아웃 주석이 그 사양이다).

harvest 이벤트의 realized 경로 의미론:
    p < accepted_len              : draft[p] 수락 → acc(draft[p])+1
    p == accepted_len < len(draft): draft[p] 기각 → rej(draft[p])+1, correction=bonus → acc(bonus)+1
    accepted_len == len(draft)    : 전량 수락 + free bonus → acc(bonus)+1
"""

from __future__ import annotations

import gzip
import json
from collections import deque
from dataclasses import dataclass, field

from core.arena import Break, SpanArena, SpanEntry, SpanProposal
from core.backoff import BackoffParams, Source, blend
from core.signature import MAX_ORDER, MIN_ORDER, RollingSigStack, fmix64, fold_key
from core.types import (
    U16_MAX,
    LedgerStats,
    Posterior,
    Segment,
    VerifyOutcome,
)

_SPAN_SALT = 0x5CA1AB1E5CA1AB1E  # span 테이블 키 공간 분리
_SNAPSHOT_SCHEMA = 1


@dataclass(frozen=True)
class StoreParams:
    version: int = 1  # 1 = hot entry만, 2 = +span arena
    k_init: int = 4
    k_max: int = 16
    coverage_target: float = 0.9
    queue_cap: int = 4096
    max_entries: int = 0  # 0 = 무제한, >0 = FIFO evict (G3 size sweep용)
    compact_every: int = 0  # 0 = off, N = drain N events마다 compaction
    decay_shift: int = 1
    orders: tuple[int, ...] = tuple(range(MIN_ORDER, MAX_ORDER + 1))
    scope_depths: tuple[int, ...] = (0, 1, 2)  # session/repo/global 활성 tier
    span_min_len: int = 6
    span_max_len: int = 64
    span_orders: tuple[int, ...] = (4, 8)  # span 등록 차수 (probe는 모든 차수)
    span_scope_depths: tuple[int, ...] = (0, 1)
    backoff: BackoffParams = field(default_factory=BackoffParams)

    @staticmethod
    def from_dict(d: dict | None) -> "StoreParams":
        if not d:
            return StoreParams()
        d = dict(d)
        if "backoff" in d:
            d["backoff"] = BackoffParams.from_dict(d["backoff"])
        for k in ("orders", "scope_depths", "span_orders", "span_scope_depths"):
            if k in d:
                d[k] = tuple(d[k])
        return StoreParams(**d)


class HotEntry:
    """cand 테이블 + 적응 k. 레이아웃 주석은 §3.1 native 사양(48–64B, k≤5 기준).

    _src는 sources_tuple() 캐시다 — 체인 확장이 harvest 사이에 같은 엔트리를 반복
    probe하므로 값어치가 있다. cands를 이 클래스 밖에서 직접 변이하는 코드는
    (compaction·white-box 테스트) 반드시 invalidate()를 호출해야 한다.
    """

    __slots__ = ("key", "seg", "dom", "scope_id", "epoch", "k_cap", "cov_ema", "cands", "_src")

    def __init__(self, key: int, seg: int, dom: int, scope_id: int, epoch: int, k_init: int):
        self.key = key
        self.seg = seg
        self.dom = dom  # epoch domain (file_id, 0=없음)
        self.scope_id = scope_id
        self.epoch = epoch
        self.k_cap = k_init
        self.cov_ema = 1.0
        # tok -> [acc u16, rej u16, logp q8 (-1=미관측)]
        self.cands: dict[int, list[int]] = {}
        self._src: tuple[tuple[int, int, int, int], ...] | None = None

    def invalidate(self) -> None:
        self._src = None

    def reset(self, epoch: int, dom: int) -> None:
        self.cands.clear()
        self.epoch = epoch
        self.dom = dom
        self.cov_ema = 1.0
        self._src = None

    def _evict_weakest(self, protect: int) -> None:
        victim, worst = None, None
        for tok, (a, r, q) in self.cands.items():
            if tok == protect:
                continue
            # 약함 = 관측 적음 > p̂ 낮음(q 큼) > tok 큼 (결정적 tie-break)
            rank = (a + r, -(q if q >= 0 else 256), -tok)
            if worst is None or rank < worst:
                worst, victim = rank, tok
        if victim is not None:
            del self.cands[victim]

    def _ensure(self, tok: int, k_max: int) -> list[int]:
        c = self.cands.get(tok)
        if c is None:
            if len(self.cands) >= self.k_cap:
                # coverage 미달이면 k 확장 (§3.1 적응 확장), 아니면 최약체 축출
                if self.cov_ema < 0.9 and self.k_cap < k_max:
                    self.k_cap = min(k_max, self.k_cap * 2)
                else:
                    self._evict_weakest(protect=tok)
            c = [0, 0, -1]
            self.cands[tok] = c
        return c

    def update_realized(self, tok: int, k_max: int) -> None:
        c = self._ensure(tok, k_max)
        if c[0] < U16_MAX:
            c[0] += 1
        self._src = None

    def update_rejected(self, tok: int, k_max: int) -> None:
        c = self._ensure(tok, k_max)
        if c[1] < U16_MAX:
            c[1] += 1
        self._src = None

    def merge_topk(self, ids: tuple[int, ...], q8s: tuple[int, ...], k_max: int) -> None:
        if not ids:
            return
        # coverage: 이번 관측 top-k 질량 중 저장분이 커버하는 비율 → EMA
        covered = sum(1 for t in ids if t in self.cands)
        cov = covered / len(ids)
        self.cov_ema = 0.75 * self.cov_ema + 0.25 * cov
        # 절단분포의 음의 증거: 이번 top-k에 없는 기존 후보의 p̂는 감쇠(+1 nat).
        # 내용 드리프트(rename 등)에서 낡은 후보가 p̂ 순위를 계속 점유하는 것을 막는다 —
        # positive-only(b)에는 없는 outcome/logit 추적 경로다.
        obs = set(ids)
        for t, c in self.cands.items():
            if t not in obs and c[2] >= 0:
                c[2] = min(255, c[2] + 16)
        for t, q in zip(ids, q8s):
            c = self.cands.get(t)
            if c is None:
                if len(self.cands) >= self.k_cap:
                    if self.cov_ema < 0.9 and self.k_cap < k_max:
                        self.k_cap = min(k_max, self.k_cap * 2)
                    else:
                        continue  # 관측 카운트 있는 기존 후보를 topk 신규가 밀어내지 않는다
                c = [0, 0, int(q)]
                self.cands[t] = c
            else:
                c[2] = int(q) if c[2] < 0 else (3 * c[2] + int(q)) >> 2
        self._src = None

    def sources_tuple(self) -> tuple[tuple[int, int, int, int], ...]:
        # q8 미관측(-1) 후보는 최저 확률로 취급 (255)
        if self._src is None:
            self._src = tuple(
                (tok, a, r, q if q >= 0 else 255) for tok, (a, r, q) in self.cands.items()
            )
        return self._src

    def total_count(self) -> int:
        return sum(a + r for a, r, _ in self.cands.values())

    def bytes(self) -> int:
        return 11 + (4 if self.dom else 0) + 9 * len(self.cands)


@dataclass
class _RunState:
    """세션별 realized run 추적 → V2 span 등록 (이벤트 경계를 넘어 이어진다)."""

    seg: int
    dom: int
    tokens: list[int]
    start_sigs: list[int]  # run 시작 시점 sig_stack (ascending order)
    scope_ids: list[int]


class LedgerStore:
    def __init__(self, params: StoreParams | None = None):
        self.params = params or StoreParams()
        self._hot: dict[int, HotEntry] = {}
        self._spans: dict[int, SpanEntry] = {}
        self._arena = SpanArena()
        self._epochs: dict[tuple[int, int], int] = {}
        self._queue: deque[VerifyOutcome] = deque()
        self._runs: dict[int, _RunState] = {}
        self._stats = LedgerStats()
        self._events_since_compact = 0

    # ------------------------------------------------------------------ epoch
    def _epoch_now(self, scope_id: int, dom: int) -> int:
        if dom == 0:
            return 0
        return self._epochs.get((scope_id, dom), 0)

    def bump_epoch(self, scope: int, file_id: int) -> None:
        """O(1). 실제 엔트리 폐기는 lazy (읽기 시 skip, compaction에서 제거)."""
        key = (scope, file_id)
        self._epochs[key] = (self._epochs.get(key, 0) + 1) & U16_MAX

    # ----------------------------------------------------------------- lookup
    def lookup(
        self, sig_stack: list[int], scope_stack: list[int], seg: Segment
    ) -> Posterior | None:
        """sig_stack[i] ↔ 차수 MIN_ORDER+i, scope_stack[d] ↔ depth d (session=0)."""
        self._stats.lookups += 1
        p = self.params
        sources: list[Source] = []
        for i, sig in enumerate(sig_stack):
            order = MIN_ORDER + i
            if order > MAX_ORDER:
                break  # 계약: sig_stack[i] ↔ 차수 MIN+i — 초과분은 무시
            if order not in p.orders:
                continue
            for depth, scope_id in enumerate(scope_stack):
                if depth not in p.scope_depths:
                    continue
                e = self._hot.get(fold_key(sig, order, scope_id, int(seg)))
                if e is None:
                    continue
                if e.epoch != self._epoch_now(e.scope_id, e.dom):
                    self._stats.stale_skips += 1
                    continue
                if e.seg != int(seg):
                    continue  # fold 충돌 가드
                sources.append(
                    Source(match_len=order, scope_depth=depth, cands=e.sources_tuple())
                )
        post = blend(p.backoff, sources)
        if post is not None:
            self._stats.hits += 1
        return post

    def lookup_span(
        self, sig_stack: list[int], scope_stack: list[int], seg: Segment
    ) -> SpanProposal | None:
        """V2: 컨텍스트에 걸린 span을 통째로 반환 (break offset 포함, §3.2)."""
        if self.params.version < 2:
            return None
        best: SpanEntry | None = None
        for i, sig in enumerate(sig_stack):
            order = MIN_ORDER + i
            if order > MAX_ORDER:
                break
            for depth, scope_id in enumerate(scope_stack):
                if depth not in self.params.scope_depths:
                    continue
                key = fmix64(fold_key(sig, order, scope_id, int(seg)) ^ _SPAN_SALT)
                s = self._spans.get(key)
                if s is None:
                    continue
                if s.epoch != self._epoch_now(s.scope_id, s.dom):
                    self._stats.stale_skips += 1
                    continue
                if s.seg != int(seg):
                    continue  # fold 충돌 가드 (hot 테이블과 동일)
                rank = (s.count, s.length, -key)
                if best is None or rank > (best.count, best.length, -best.key):
                    best = s
        if best is None:
            return None
        toks = self._arena.get(best.arena_off, best.length)
        brk = tuple(sorted((off, b.count) for off, b in best.breaks.items()))
        return SpanProposal(tokens=toks, breaks=brk, count=best.count, key=best.key)

    # ---------------------------------------------------------------- harvest
    def harvest(self, events: list[VerifyOutcome]) -> None:
        """enqueue만 한다 — 절대 블로킹 금지. 포화 시 drop-oldest + 카운터 (§3.1).

        queue_cap <= 0 은 '큐 비활성' — 모든 이벤트를 드롭하고 카운트만 남긴다.
        """
        q = self._queue
        cap = self.params.queue_cap
        if cap <= 0:
            self._stats.drops += len(events)
            return
        for e in events:
            if len(q) >= cap:
                q.popleft()
                self._stats.drops += 1
            q.append(e)

    def queue_depth(self) -> int:
        return len(self._queue)

    def drain(self, max_events: int | None = None) -> int:
        """큐를 소비해 실제 반영. online은 background thread, sim은 step 동기 호출."""
        n = 0
        while self._queue and (max_events is None or n < max_events):
            self._apply(self._queue.popleft())
            n += 1
        if n:
            self._events_since_compact += n
            ce = self.params.compact_every
            if ce > 0 and self._events_since_compact >= ce:
                self.compact()
                self._events_since_compact = 0
        return n

    def _get_or_create(self, key: int, seg: int, dom: int, scope_id: int) -> HotEntry:
        e = self._hot.get(key)
        now = self._epoch_now(scope_id, dom)
        if e is None:
            p = self.params
            if p.max_entries and len(self._hot) >= p.max_entries:
                oldest = next(iter(self._hot))
                del self._hot[oldest]
                self._stats.evictions += 1
            e = HotEntry(key, seg, dom, scope_id, now, p.k_init)
            self._hot[key] = e
        elif e.epoch != self._epoch_now(e.scope_id, e.dom):
            e.reset(now, dom)  # stale 엔트리 재사용: write 시점 재검증
        elif dom and e.dom != dom:
            # epoch domain은 '최근 관측'의 파일을 따른다 — 최초-기록자 고착이면
            # 나중 파일의 편집이 이 엔트리를 영영 무효화하지 못한다 (stale 위험)
            e.dom = dom
            e.epoch = now
        return e

    def _apply(self, ev: VerifyOutcome) -> None:
        p = self.params
        scope_ids = [sid for _, sid in ev.scope.scope_stack()]
        sigs = RollingSigStack()
        sigs.push_many(ev.ctx_tail)

        realized = ev.realized()
        draft_len = len(ev.draft_ids)
        seg_arr = ev.seg

        # V2: 이 draft가 기존 span에서 나온 프리픽스라면 break 기록 준비
        span_at_start: SpanEntry | None = None
        if p.version >= 2 and draft_len:
            seg0 = seg_arr[0] if seg_arr else int(Segment.TEXT)
            span_at_start = self._find_span_entry(sigs.stack_list(), scope_ids, seg0)

        patch_key = 0
        patch_rank = (-1, -1)
        for pos in range(len(realized)):
            seg_p = seg_arr[min(pos, len(seg_arr) - 1)] if seg_arr else int(Segment.TEXT)
            dom = ev.file_id if seg_p == int(Segment.CODE) else 0
            tok = realized[pos]
            is_reject_pos = pos == ev.accepted_len and pos < draft_len

            stack = sigs.stack_list()
            for i, sig in enumerate(stack):
                order = MIN_ORDER + i
                if order not in p.orders:
                    continue
                for depth, sid in enumerate(scope_ids):
                    if depth not in p.scope_depths:
                        continue
                    key = fold_key(sig, order, sid, seg_p)
                    e = self._get_or_create(key, seg_p, dom, sid)
                    e.update_realized(tok, p.k_max)
                    if is_reject_pos:
                        e.update_rejected(ev.draft_ids[pos], p.k_max)
                    if pos < len(ev.topk_ids):
                        e.merge_topk(ev.topk_ids[pos], ev.topk_logp_q8[pos], p.k_max)
                    if is_reject_pos and (order, -depth) > patch_rank:
                        # 실제 probe된 조합 중 최장 차수·최심 tier의 correction 엔트리
                        patch_rank = (order, -depth)
                        patch_key = key

            if p.version >= 2:
                self._track_run(ev, sigs, seg_p, dom, tok, scope_ids)
            sigs.push(tok)

        if (
            span_at_start is not None
            and ev.accepted_len < draft_len
            and ev.accepted_len < span_at_start.length
            and self._arena.get(span_at_start.arena_off, min(draft_len, span_at_start.length))[
                : ev.accepted_len + 1
            ]
            == tuple(ev.draft_ids[: ev.accepted_len + 1])
        ):
            b = span_at_start.breaks.setdefault(ev.accepted_len, Break())
            b.count += 1
            if patch_key:
                b.patch_key = patch_key
        elif span_at_start is not None and ev.accepted_len == draft_len:
            span_at_start.count += 1  # span 재사용 성공 신호

        self._stats.harvested_events += 1
        self._refresh_sizes()

    def _find_span_entry(
        self, sig_stack: list[int], scope_ids: list[int], seg: int
    ) -> SpanEntry | None:
        best: SpanEntry | None = None
        for i, sig in enumerate(sig_stack):
            order = MIN_ORDER + i
            if order > MAX_ORDER:
                break
            for depth, scope_id in enumerate(scope_ids):
                if depth not in self.params.scope_depths:
                    continue
                key = fmix64(fold_key(sig, order, scope_id, seg) ^ _SPAN_SALT)
                s = self._spans.get(key)
                if (
                    s is not None
                    and s.epoch == self._epoch_now(s.scope_id, s.dom)
                    and s.seg == seg
                ):
                    # lookup_span과 동일한 랭킹 — break 기록과 조회가 같은 엔트리를 보도록
                    if best is None or (s.count, s.length, -s.key) > (
                        best.count,
                        best.length,
                        -best.key,
                    ):
                        best = s
        return best

    # -------------------------------------------------------------- V2 spans
    def _track_run(
        self,
        ev: VerifyOutcome,
        sigs: RollingSigStack,
        seg: int,
        dom: int,
        tok: int,
        scope_ids: list[int],
    ) -> None:
        sid = ev.scope.session_id()
        run = self._runs.get(sid)
        if run is not None and (run.seg != seg or len(run.tokens) >= self.params.span_max_len):
            self._flush_run(sid)
            run = None
        if run is None:
            run = _RunState(
                seg=seg, dom=dom, tokens=[], start_sigs=sigs.stack_list(), scope_ids=scope_ids
            )
            self._runs[sid] = run
        run.tokens.append(tok)
        if dom:
            run.dom = dom

    def _flush_run(self, sid: int) -> None:
        run = self._runs.pop(sid, None)
        if run is None or len(run.tokens) < self.params.span_min_len:
            return
        off, length = self._arena.add(run.tokens)
        for i, sig in enumerate(run.start_sigs):
            order = MIN_ORDER + i
            if order not in self.params.span_orders:
                continue
            for depth, scope_id in enumerate(run.scope_ids):
                if depth not in self.params.span_scope_depths:
                    continue
                key = fmix64(fold_key(sig, order, scope_id, run.seg) ^ _SPAN_SALT)
                s = self._spans.get(key)
                if s is None:
                    self._spans[key] = SpanEntry(
                        key=key,
                        arena_off=off,
                        length=length,
                        scope_id=scope_id,
                        seg=run.seg,
                        dom=run.dom,
                        epoch=self._epoch_now(scope_id, run.dom),
                        count=1,
                    )
                else:
                    if s.epoch != self._epoch_now(s.scope_id, s.dom):
                        s.breaks.clear()
                        s.epoch = self._epoch_now(scope_id, run.dom)
                        s.dom = run.dom
                        s.count = 0
                    # 더 길거나 최신 관측이면 span 본문 갱신. break 히스토그램은 내용
                    # 동일성에 종속 — 새 내용과 갈라지는 지점(d) 이후의 break는 무효
                    if length >= s.length:
                        if s.arena_off != off:
                            old = self._arena.get(s.arena_off, s.length)
                            d = 0
                            for d in range(min(len(old), length)):  # noqa: B007
                                if old[d] != run.tokens[d]:
                                    break
                            else:
                                d = min(len(old), length)
                            for boff in [b for b in s.breaks if b >= d]:
                                del s.breaks[boff]
                        s.arena_off, s.length = off, length
                    s.count += 1

    def flush_runs(self) -> None:
        """세션 종료/요청 종료 시 잔여 run을 span으로 확정 (sim replay가 호출)."""
        for sid in list(self._runs):
            self._flush_run(sid)

    # ------------------------------------------------------------- compaction
    def compact(self) -> None:
        p = self.params
        live_hot: dict[int, HotEntry] = {}
        for key, e in self._hot.items():
            if e.epoch != self._epoch_now(e.scope_id, e.dom):
                continue  # lazy 폐기 실행 지점 (§3.1)
            dead = []
            for tok, c in e.cands.items():
                c[0] >>= p.decay_shift
                c[1] >>= p.decay_shift
                if c[0] == 0 and c[1] == 0:
                    dead.append(tok)
            for tok in dead:
                del e.cands[tok]
            e.invalidate()  # cands 직접 변이 — sources 캐시 무효화
            if e.cands:
                e.k_cap = max(p.k_init, min(e.k_cap, p.k_max))
                live_hot[key] = e
        self._hot = live_hot

        live_spans: dict[int, SpanEntry] = {}
        for key, s in self._spans.items():
            if s.epoch != self._epoch_now(s.scope_id, s.dom):
                continue
            s.count >>= p.decay_shift
            if s.count <= 0:
                continue
            dead_b = []
            for off, b in s.breaks.items():
                b.count >>= p.decay_shift
                if b.count <= 0:
                    dead_b.append(off)
            for off in dead_b:
                del s.breaks[off]
            live_spans[key] = s
        self._spans = live_spans
        self._arena.compact(list(self._spans.values()))
        self._stats.compactions += 1
        self._refresh_sizes()

    # ----------------------------------------------------------- persistence
    def snapshot(self, path: str) -> None:
        """영속 상태만 저장한다: hot/span 테이블, arena, epoch 테이블.

        저장하지 않는 것: stats 카운터(관측치), harvest 큐(미반영 이벤트 —
        호출측이 drain 후 snapshot해야 한다), run 버퍼(flush_runs로 확정 후 저장).
        cov_ema는 전체 정밀도로 저장한다 — 반올림하면 load 후 이어가기가
        무중단 실행과 어긋난다.
        """
        doc = {
            "schema": _SNAPSHOT_SCHEMA,
            "version": self.params.version,
            "hot": [
                {
                    "key": e.key,
                    "seg": e.seg,
                    "dom": e.dom,
                    "scope_id": e.scope_id,
                    "epoch": e.epoch,
                    "k_cap": e.k_cap,
                    "cov": e.cov_ema,  # 전체 정밀도 (json float repr는 정확 왕복)
                    "cands": {str(t): c for t, c in e.cands.items()},
                }
                for e in self._hot.values()
            ],
            "spans": [
                {
                    "key": s.key,
                    "off": s.arena_off,
                    "len": s.length,
                    "scope_id": s.scope_id,
                    "seg": s.seg,
                    "dom": s.dom,
                    "epoch": s.epoch,
                    "count": s.count,
                    "breaks": {
                        str(o): [b.count, b.patch_key] for o, b in s.breaks.items()
                    },
                }
                for s in self._spans.values()
            ],
            "arena": self._arena.dump(),
            "epochs": [[k[0], k[1], v] for k, v in self._epochs.items()],
        }
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(doc, f, sort_keys=True, separators=(",", ":"))

    def load(self, path: str) -> None:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["schema"] == _SNAPSHOT_SCHEMA, f"snapshot schema {doc['schema']} 미지원"
        self._hot = {}
        for h in doc["hot"]:
            e = HotEntry(
                h["key"], h["seg"], h["dom"], h["scope_id"], h["epoch"], self.params.k_init
            )
            e.k_cap = h["k_cap"]
            e.cov_ema = h["cov"]
            e.cands = {int(t): list(c) for t, c in h["cands"].items()}
            self._hot[e.key] = e
        self._spans = {}
        for sd in doc["spans"]:
            s = SpanEntry(
                key=sd["key"],
                arena_off=sd["off"],
                length=sd["len"],
                scope_id=sd["scope_id"],
                seg=sd["seg"],
                dom=sd["dom"],
                epoch=sd["epoch"],
                count=sd["count"],
                breaks={int(o): Break(c[0], c[1]) for o, c in sd["breaks"].items()},
            )
            self._spans[s.key] = s
        self._arena = SpanArena()
        self._arena.load(doc["arena"])
        self._epochs = {(k[0], k[1]): k[2] for k in doc["epochs"]}
        self._refresh_sizes()

    # ----------------------------------------------------------------- stats
    def _refresh_sizes(self) -> None:
        st = self._stats
        st.entries = len(self._hot)
        st.span_entries = len(self._spans)
        st.arena_tokens = self._arena.n_tokens()
        st.queue_depth = len(self._queue)

    def stats(self) -> LedgerStats:
        self._refresh_sizes()
        st = self._stats
        st.bytes = (
            sum(e.bytes() for e in self._hot.values())
            + sum(s.bytes() for s in self._spans.values())
            + self._arena.bytes()
        )
        return st
