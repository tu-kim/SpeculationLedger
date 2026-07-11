"""core.backoff: λ 단조성과 blend 성질."""

from core.backoff import BackoffParams, Source, blend, lam
from core.signature import MAX_ORDER

P = BackoffParams()


def test_lambda_monotone_in_match_len():
    for d in (0, 1, 2):
        vals = [lam(P, o, d, 5) for o in range(2, MAX_ORDER + 1)]
        assert vals == sorted(vals), "match_len 증가 시 λ 비감소여야 함"


def test_lambda_monotone_in_scope_depth():
    vals = [lam(P, 6, d, 5) for d in (0, 1, 2)]
    assert vals == sorted(vals, reverse=True), "depth 증가(일반화) 시 λ 비증가여야 함"


def test_lambda_monotone_in_count():
    vals = [lam(P, 6, 0, c) for c in (1, 2, 8, 64)]
    assert vals == sorted(vals), "count 증가 시 λ 비감소여야 함"


def test_blend_empty_returns_none():
    assert blend(P, []) is None


def test_blend_ranks_frequent_token_at_low_order():
    # 저차수(앨리어싱) 소스만: p̂/rej는 무시되고 acc-빈도가 랭킹을 지배해야 한다
    src = Source(match_len=2, scope_depth=0, cands=(
        (111, 5, 0, 200),  # 빈도 높음, p̂ 나쁨
        (222, 1, 0, 0),    # 빈도 낮음, p̂ 최고
    ))
    post = blend(P, [src])
    assert post.cands[0].tok == 111


def test_blend_trusts_p_hat_at_high_order():
    # 고차수 소스: 같은 빈도면 p̂가 랭킹을 결정
    src = Source(match_len=8, scope_depth=0, cands=(
        (111, 3, 3, 96),  # 기각 많고 p̂ 낮음
        (222, 3, 0, 2),   # p̂ 높음
    ))
    post = blend(P, [src])
    assert post.cands[0].tok == 222
    # p_acc: 고차수 기각 증거가 게이트를 통과해 반영됨
    c111 = next(c for c in post.cands if c.tok == 111)
    c222 = next(c for c in post.cands if c.tok == 222)
    assert c111.p_acc < c222.p_acc


def test_blend_session_outweighs_global():
    sess = Source(match_len=5, scope_depth=0, cands=((111, 4, 0, 8),))
    glob = Source(match_len=5, scope_depth=2, cands=((222, 4, 0, 8),))
    post = blend(P, [sess, glob])
    assert post.cands[0].tok == 111
