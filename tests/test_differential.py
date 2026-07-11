"""차분 검증: LedgerStore vs 참조 NaiveStore.

NaiveStore는 fold_key 해싱 없이 (차수, scope_id, seg, 정확한 컨텍스트 튜플)로 키잉하고
동일한 HotEntry 산술·blend를 사용한다. 무작위 워크로드에서 두 구현의 posterior가
완전 일치해야 한다 — 다음을 한 번에 검증한다:
  - fold_key가 이 워크로드에서 무충돌 (충돌 시 엔트리 병합 → posterior 불일치)
  - _apply 위치 워크·차수/scope 열거 순서의 정합
  - epoch(bump) 의미론의 정합
  - RollingSigStack 서명 ↔ 정확 컨텍스트의 1:1 대응
"""

import random

from core.backoff import Source, blend
from core.signature import MAX_ORDER, MIN_ORDER, RollingSigStack
from core.store import HotEntry, LedgerStore, StoreParams
from core.types import U16_MAX, Scope, Segment, VerifyOutcome


class NaiveStore:
    """참조 구현: 해싱 없는 정확-컨텍스트 키잉. 산술은 HotEntry/blend 공유."""

    def __init__(self, params: StoreParams):
        self.p = params
        self.tab: dict[tuple, HotEntry] = {}
        self.epochs: dict[tuple[int, int], int] = {}

    def _now(self, scope_id: int, dom: int) -> int:
        return self.epochs.get((scope_id, dom), 0) if dom else 0

    def bump_epoch(self, scope_id: int, file_id: int) -> None:
        k = (scope_id, file_id)
        self.epochs[k] = (self.epochs.get(k, 0) + 1) & U16_MAX

    def apply(self, ev: VerifyOutcome) -> None:
        prefix = list(ev.ctx_tail)
        realized = ev.realized()
        draft_len = len(ev.draft_ids)
        scope_ids = [sid for _, sid in ev.scope.scope_stack()]
        for pos, tok in enumerate(realized):
            seg_p = ev.seg[min(pos, len(ev.seg) - 1)] if ev.seg else int(Segment.TEXT)
            dom = ev.file_id if seg_p == int(Segment.CODE) else 0
            is_rej = pos == ev.accepted_len and pos < draft_len
            hi = min(len(prefix), MAX_ORDER)
            for order in range(MIN_ORDER, hi + 1):
                if order not in self.p.orders:
                    continue
                ctx = tuple(prefix[-order:])
                for depth, sid in enumerate(scope_ids):
                    if depth not in self.p.scope_depths:
                        continue
                    key = (order, sid, seg_p, ctx)
                    e = self.tab.get(key)
                    now = self._now(sid, dom)
                    if e is None:
                        e = HotEntry(0, seg_p, dom, sid, now, self.p.k_init)
                        self.tab[key] = e
                    elif e.epoch != self._now(e.scope_id, e.dom):
                        e.reset(now, dom)
                    elif dom and e.dom != dom:
                        e.dom = dom
                        e.epoch = now
                    e.update_realized(tok, self.p.k_max)
                    if is_rej:
                        e.update_rejected(ev.draft_ids[pos], self.p.k_max)
                    if pos < len(ev.topk_ids):
                        e.merge_topk(ev.topk_ids[pos], ev.topk_logp_q8[pos], self.p.k_max)
            prefix.append(tok)

    def lookup(self, ctx: list[int], scope_ids: list[int], seg: int):
        sources = []
        hi = min(len(ctx), MAX_ORDER)
        for order in range(MIN_ORDER, hi + 1):
            if order not in self.p.orders:
                continue
            key_ctx = tuple(ctx[-order:])
            for depth, sid in enumerate(scope_ids):
                if depth not in self.p.scope_depths:
                    continue
                e = self.tab.get((order, sid, seg, key_ctx))
                if e is None:
                    continue
                if e.epoch != self._now(e.scope_id, e.dom):
                    continue
                if e.seg != seg:
                    continue
                sources.append(Source(match_len=order, scope_depth=depth, cands=e.sources_tuple()))
        return blend(self.p.backoff, sources)


def _next_tokens(rng: random.Random, stream: list[int], vocab: int, n: int) -> list[int]:
    """재발 구조가 있는 다음 토큰열: 70%는 과거 구간 재방출, 30%는 신규."""
    out: list[int] = []
    while len(out) < n:
        if len(stream) > 16 and rng.random() < 0.7:
            j = rng.randrange(0, len(stream) - 8)
            take = rng.randint(2, 6)
            out.extend(stream[j : j + take])
        else:
            out.append(rng.randrange(16, vocab))
    return out[:n]


def _rand_event(rng: random.Random, scope: Scope, stream: list[int], vocab: int):
    """stream 꼬리를 컨텍스트로 하는 verify outcome. realized는 재발 구조를 갖는다
    (과거 구간 재방출) — lookup 대조가 비-None posterior에서 이뤄지도록."""
    ctx_tail = tuple(stream[-MAX_ORDER:])
    acc = rng.randint(0, 4)
    realized = _next_tokens(rng, stream, vocab, acc + 1)
    # draft = accepted prefix (+기각 후보 1개, 절반 확률)
    draft = list(realized[:acc])
    if rng.random() < 0.5:
        draft.append(rng.randrange(16, vocab))
    n_real = acc + 1
    seg_val = rng.randrange(0, 4)  # step 내 단일 seg — 실제 스트림과 유사
    seg = tuple(seg_val for _ in range(n_real))
    file_id = rng.choice([0, 7, 9])
    if rng.random() < 0.7:
        topk_ids = tuple(
            (realized[i],) + tuple(rng.randrange(16, vocab) for _ in range(rng.randint(0, 3)))
            for i in range(n_real)
        )
        topk_q8 = tuple(tuple(rng.randrange(0, 200) for _ in row) for row in topk_ids)
    else:
        topk_ids = tuple(() for _ in range(n_real))
        topk_q8 = tuple(() for _ in range(n_real))
    ev = VerifyOutcome(
        scope=scope,
        ctx_tail=ctx_tail,
        draft_ids=tuple(draft),
        accepted_len=acc,
        bonus_id=realized[-1],
        topk_ids=topk_ids,
        topk_logp_q8=topk_q8,
        seg=seg,
        file_id=file_id,
    )
    stream.extend(realized)
    return ev


def _posterior_repr(p):
    if p is None:
        return None
    return [
        (c.tok, round(c.p_acc, 12), round(c.p_hat, 12), c.support) for c in p.cands
    ] + [round(p.weight, 12), p.best_order]


def test_differential_store_vs_naive_reference():
    params = StoreParams(version=1)  # 차분 대상은 hot 경로 (span은 별도 테스트)
    vocab = 120  # 작은 어휘 → 문맥 재발 풍부 → 엔트리 재사용·충돌 압력 극대화
    n_cmp, n_hit = [0], [0]  # 테스트 자체의 민감도 계측

    for trial in range(4):
        rng = random.Random(1000 + trial)
        scopes = [
            Scope("tA", "tA/r1", "tA/r1/s1"),
            Scope("tA", "tA/r1", "tA/r1/s2"),  # repo 공유
            Scope("tB", "tB/r2", "tB/r2/s1"),  # tenant 분리
        ]
        real = LedgerStore(params)
        naive = NaiveStore(params)
        streams = {s.session: [rng.randrange(16, vocab) for _ in range(MAX_ORDER)] for s in scopes}

        for step in range(300):
            scope = scopes[rng.randrange(len(scopes))]
            ev = _rand_event(rng, scope, streams[scope.session], vocab)
            real.harvest([ev])
            real.drain()
            naive.apply(ev)

            if rng.random() < 0.05:  # 무작위 invalidation
                sc = scopes[rng.randrange(len(scopes))]
                fid = rng.choice([7, 9])
                for sid in (sc.session_id(), sc.repo_id()):
                    real.bump_epoch(sid, fid)
                    naive.bump_epoch(sid, fid)

            if step % 7 == 0:
                # 관측 스트림의 무작위 절단점 + 신선한 컨텍스트 양쪽에서, 4개 seg 전부 대조
                for _ in range(4):
                    sc = scopes[rng.randrange(len(scopes))]
                    st = streams[sc.session]
                    if rng.random() < 0.8:
                        end = rng.randint(MIN_ORDER, len(st))
                        ctx = st[max(0, end - 24) : end]  # 과거 절단점 → 재발 문맥 적중
                    else:
                        ctx = [rng.randrange(16, vocab) for _ in range(rng.randint(2, 10))]
                    scope_ids = [sid for _, sid in sc.scope_stack()]
                    rs = RollingSigStack()
                    rs.push_many(ctx)
                    sig_stack = rs.stack_list()
                    for seg in range(4):
                        got = real.lookup(sig_stack, scope_ids, Segment(seg))
                        want = naive.lookup(ctx, scope_ids, seg)
                        n_cmp[0] += 1
                        n_hit[0] += got is not None
                        assert _posterior_repr(got) == _posterior_repr(want), (
                            f"trial {trial} step {step} seg {seg}: posterior 불일치\n"
                            f"ctx={ctx}\n got={_posterior_repr(got)}\nwant={_posterior_repr(want)}"
                        )

        # 키 공간 정합: fold 충돌이 있었다면 real 쪽 엔트리가 더 적다
        assert real.stats().entries == len(naive.tab), "fold_key 충돌 또는 워크 불일치"

    # 민감도 자가 검증: 비교의 상당수가 실제(비-None) posterior 대조여야 한다.
    # 이 비율이 무너지면 테스트가 None==None만 세는 눈먼 상태다 — 즉시 실패시킨다.
    assert n_hit[0] / n_cmp[0] >= 0.15, f"차분 테스트 둔감: hit {n_hit[0]}/{n_cmp[0]}"


def test_differential_epoch_semantics_after_wrap_free_bumps():
    """bump가 잦아도 두 구현의 stale 판정이 동일해야 한다 (code seg 집중 워크로드)."""
    params = StoreParams(version=1)
    rng = random.Random(77)
    scope = Scope("t", "t/r", "t/r/s")
    real, naive = LedgerStore(params), NaiveStore(params)
    stream = [rng.randrange(16, 80) for _ in range(MAX_ORDER)]

    for step in range(200):
        ctx_tail = tuple(stream[-MAX_ORDER:])
        bonus = _next_tokens(rng, stream, 80, 1)[0]  # 재발 구조 유지
        ev = VerifyOutcome(
            scope=scope,
            ctx_tail=ctx_tail,
            draft_ids=(),
            accepted_len=0,
            bonus_id=bonus,
            topk_ids=((),),
            topk_logp_q8=((),),
            seg=(int(Segment.CODE),),
            file_id=rng.choice([5, 6]),
        )
        stream.append(bonus)
        real.harvest([ev])
        real.drain()
        naive.apply(ev)
        if rng.random() < 0.2:
            fid = rng.choice([5, 6])
            for sid in (scope.session_id(), scope.repo_id()):
                real.bump_epoch(sid, fid)
                naive.bump_epoch(sid, fid)

        ctx = stream[-rng.randint(MIN_ORDER, 16) :]
        rs = RollingSigStack()
        rs.push_many(ctx)
        scope_ids = [sid for _, sid in scope.scope_stack()]
        got = real.lookup(rs.stack_list(), scope_ids, Segment.CODE)
        want = naive.lookup(ctx, scope_ids, int(Segment.CODE))
        assert _posterior_repr(got) == _posterior_repr(want), f"step {step} epoch 불일치"
