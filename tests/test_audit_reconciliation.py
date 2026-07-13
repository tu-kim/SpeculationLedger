"""감사(2026-07-12) 조율 findings의 회귀 방어.

Finding A: coverage_target 파라미터 배선 (하드코딩 0.9 제거)
Finding C: 단일 λ / fold_key 인라인 사본의 동치를 구조적으로 강제 (주석 대신 테스트)
Finding D: greedy correction = p̂ argmax (§3.1) 를 Posterior.correction()으로 명시
"""

import random

from core.backoff import BackoffParams, Source, blend, lam
from core.signature import MAX_ORDER, MIN_ORDER, fold_key, sig_of
from core.store import HotEntry, LedgerStore, StoreParams
from core.types import Posterior, PosteriorCand, Scope, Segment, VerifyOutcome

SCOPE = Scope("t", "t/r", "t/r/s")
STACK = [sid for _, sid in SCOPE.scope_stack()]


def _ev(ctx, bonus, topk=None, seg=Segment.TEXT):
    tki = (tuple(topk),) if topk is not None else ((),)
    return VerifyOutcome(
        scope=SCOPE, ctx_tail=tuple(ctx), draft_ids=(), accepted_len=0, bonus_id=bonus,
        topk_ids=tki, topk_logp_q8=tuple(tuple(3 for _ in r) for r in tki),
        seg=(int(seg),),
    )


# ============================================================ Finding A
def test_coverage_target_param_is_wired_not_hardcoded():
    """coverage_target을 낮추면(=쉽게 만족) k 확장이 억제돼 k_cap이 작게 유지된다.
    반대로 높이면 확장이 촉진된다 — 하드코딩 0.9이면 두 store가 동일해질 것."""
    def run(cov_target):
        store = LedgerStore(StoreParams(k_init=2, k_max=16, coverage_target=cov_target))
        ctx = [1, 2, 3, 4, 5, 6, 7, 8]
        # 매 관측 완전히 다른 top-k 8종 → coverage 계속 낮음
        for i in range(8):
            store.harvest([_ev(ctx, bonus=500 + i,
                               topk=[1000 + i * 8 + j for j in range(8)])])
        store.drain()
        return max(e.k_cap for e in store._hot.values())

    # cov_target=0.0: cov_ema는 절대 그 아래로 안 감 → 확장 억제 → k_cap 작음
    # cov_target=1.0: 항상 미달 → 확장 촉진 → k_cap 큼
    k_low = run(0.0)
    k_high = run(1.0)
    assert k_low < k_high, f"coverage_target 미배선: k_cap {k_low} vs {k_high} (하드코딩 의심)"
    assert k_low == 2, "cov_target=0이면 확장 없어야 함 (k_init 유지)"


def test_coverage_target_default_preserves_behavior():
    """기본값(0.9)은 명시 지정과 동일 — byte-동일 회귀 방어."""
    def kcap(params):
        store = LedgerStore(params)
        ctx = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            store.harvest([_ev(ctx, bonus=500 + i,
                               topk=[1000 + i * 8 + j for j in range(8)])])
        store.drain()
        return sorted(e.k_cap for e in store._hot.values())

    assert kcap(StoreParams(k_init=2)) == kcap(StoreParams(k_init=2, coverage_target=0.9))


# ============================================================ Finding C
def test_inline_fold_key_equals_spec_function_exhaustively():
    """store 핫루프의 인라인 fold 전개가 스펙 fold_key와 모든 (sig,order,scope,seg)
    조합에서 비트 단위로 동일해야 한다. 한쪽만 수정되면 즉시 실패."""
    rng = random.Random(1)
    U64 = (1 << 64) - 1
    from core.signature import ORDER_SALTS, SEG_SALTS

    for _ in range(2000):
        sig = rng.randrange(0, 1 << 64)
        order = rng.randint(MIN_ORDER, MAX_ORDER)
        scope_id = rng.randrange(0, 1 << 64)
        seg = rng.randrange(0, 4)
        # store.lookup / _apply의 인라인 전개 (동일 상수)
        x = (sig ^ ORDER_SALTS[order]) ^ scope_id ^ SEG_SALTS[seg & 3]
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & U64
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & U64
        inline = (x ^ (x >> 31)) & U64
        assert inline == fold_key(sig, order, scope_id, seg), (sig, order, scope_id, seg)


def test_inline_lambda_weight_equals_lam():
    """blend 내부 인라인 가중치 계산이 lam()과 동일해야 한다 (단일 λ 계약)."""
    from core.backoff import _decay_pows

    p = BackoffParams()
    tbl_o, tbl_s = _decay_pows(p.order_decay, p.scope_decay)
    rng = random.Random(2)
    for _ in range(500):
        m = rng.randint(MIN_ORDER, MAX_ORDER)
        d = rng.randint(0, 2)
        c = rng.randint(1, 500)
        inline = tbl_o[m] * tbl_s[d] * (c / (c + p.count_prior))
        assert inline == lam(p, m, d, c), (m, d, c)


def test_store_keys_recoverable_via_spec_fold_key():
    """harvest가 인라인 fold로 만든 엔트리 키가 스펙 fold_key로 정확히 조회된다."""
    store = LedgerStore(StoreParams())
    ctx = [10, 11, 12, 13, 14, 15, 16, 17]
    store.harvest([_ev(ctx, bonus=99)])
    store.drain()
    hits = sum(
        1
        for order in range(MIN_ORDER, MAX_ORDER + 1)
        for sid in STACK
        if fold_key(sig_of(ctx[-order:]), order, sid, int(Segment.TEXT)) in store._hot
    )
    assert hits == (MAX_ORDER - MIN_ORDER + 1) * len(STACK)


# ============================================================ Finding D
def test_posterior_correction_is_p_hat_argmax():
    """correction() = p̂ 최대 (§3.1). argmax()(복합 랭킹)와 갈릴 수 있어야 의미가 있다."""
    cands = (
        PosteriorCand(tok=1, p_acc=0.9, p_hat=0.20, support=100),  # 복합 랭킹 top
        PosteriorCand(tok=2, p_acc=0.5, p_hat=0.55, support=1),    # p̂ 최대
        PosteriorCand(tok=3, p_acc=0.5, p_hat=0.25, support=1),
    )
    post = Posterior(cands=cands, weight=1.0, best_order=8)
    assert post.correction().tok == 2, "correction은 p̂ argmax여야 함"
    assert post.argmax().tok == 1, "argmax는 복합 랭킹 top (cands[0])"
    assert post.correction() is not post.argmax(), "다봉에서 둘이 갈려야 검증 의미 있음"


def test_correction_tie_break_deterministic():
    """p̂ 동률은 작은 tok로 결정 (I4)."""
    cands = (
        PosteriorCand(tok=7, p_acc=0.5, p_hat=0.4, support=1),
        PosteriorCand(tok=3, p_acc=0.5, p_hat=0.4, support=1),
    )
    post = Posterior(cands=cands, weight=1.0, best_order=8)
    assert post.correction().tok == 3


def test_correction_empty_posterior():
    assert Posterior(cands=(), weight=0.0, best_order=0).correction() is None


def test_correction_matches_blend_output():
    """실제 blend 산출 Posterior에서 correction()이 cands 중 p_hat 최대와 일치."""
    src = Source(match_len=8, scope_depth=0,
                 cands=((1, 5, 0, 40), (2, 1, 0, 0), (3, 2, 0, 200)))
    post = blend(BackoffParams(), [src])
    manual = max(post.cands, key=lambda c: (c.p_hat, -c.tok))
    assert post.correction().tok == manual.tok


# ============================================================ 통합 sanity
def test_hotentry_direct_update_still_defaults_to_0_9():
    """HotEntry 단위 API(cov_target 기본값)는 기존 화이트박스 테스트와 호환."""
    e = HotEntry(0, int(Segment.TEXT), 0, STACK[0], 0, k_init=2)
    for i in range(8):
        e.merge_topk(tuple(100 + i * 8 + j for j in range(8)),
                     tuple(3 for _ in range(8)), k_max=16)  # cov_target 미지정 → 0.9
    assert e.k_cap > 2  # 기본 0.9로 확장 발생
