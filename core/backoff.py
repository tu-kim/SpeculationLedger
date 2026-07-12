"""이중 back-off: 차수 2..8 × scope(session→repo→global)를 단일 보간
λ(match_len, scope_depth, count)로 통합한다 (CLAUDE.md §3.1).

λ = order_decay^(MAX_ORDER - match_len) · scope_decay^scope_depth · count/(count + count_prior)

- match_len(=차수)이 길수록 신뢰↑ (order_decay < 1)
- scope가 구체적일수록(session=depth0) 신뢰↑ (scope_decay < 1)
- 관측 count가 클수록 신뢰↑ (shrinkage)

blend()는 (source, HotEntry 후보들)의 가중 혼합으로 단일 Posterior를 만든다.
p_acc는 Beta(0.5,0.5) Jeffreys smoothing된 acc/(acc+rej)의 가중 평균,
p_hat은 q8 로그확률의 가중 확률 평균을 소스 간 정규화한 값이다.
correction 분포는 별도 저장 없이 p_hat 그 자체다 (§3.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

from core.signature import MAX_ORDER
from core.types import Posterior, PosteriorCand, q8_to_p


@dataclass(frozen=True)
class BackoffParams:
    order_decay: float = 0.55  # 차수 1 감소당 가중 감쇠
    scope_decay: float = 0.35  # scope 1단계 일반화당 감쇠
    count_prior: float = 2.0  # count shrinkage 의사 카운트 κ
    beta_a: float = 0.5  # p_acc Jeffreys prior
    beta_b: float = 0.5
    max_cands: int = 8  # blend 결과로 유지할 후보 수
    # annotation(rej·p̂) 신뢰 게이트. 앨리어싱은 '차수'의 속성이다: 저차수 키에는
    # 다수 문맥이 겹쳐 rej 카운트와 p̂(EMA)가 노이즈가 된다. 따라서 annotation은
    # match_len ≥ p_hat_min_order 소스에서만 블렌드하고(freq는 전 차수),
    # 그 질량에 gate = w/(w+prior_w)를 곱해 반영한다.
    anno_prior_w: float = 0.25
    p_hat_min_order: int = 4

    @staticmethod
    def from_dict(d: dict | None) -> "BackoffParams":
        return BackoffParams(**d) if d else BackoffParams()


@lru_cache(maxsize=32)
def _decay_pows(
    order_decay: float, scope_decay: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """pow LUT — lookup마다 소스별 float 거듭제곱을 상수화. 값은 직접 계산과 동일(I4)."""
    return (
        tuple(order_decay ** (MAX_ORDER - n) for n in range(MAX_ORDER + 1)),
        tuple(scope_decay**d for d in range(8)),
    )


def lam(params: BackoffParams, match_len: int, scope_depth: int, count: int) -> float:
    """단일 보간 λ. 세 인자 모두에 단조: match_len↑ ⇒ λ↑, depth↑ ⇒ λ↓, count↑ ⇒ λ↑."""
    tbl_o, tbl_s = _decay_pows(params.order_decay, params.scope_decay)
    return tbl_o[match_len] * tbl_s[scope_depth] * (count / (count + params.count_prior))


class Source(NamedTuple):
    """blend 입력 1건 = (어느 차수/scope에서 매치됐나, 후보 통계).

    total/total_acc는 store의 엔트리 캐시가 채우는 사전계산치다 — 미지정(-1)이면
    cands에서 재계산하므로 직접 구성하는 테스트·참조 구현과 값이 동일하다.
    """

    match_len: int
    scope_depth: int
    # 후보별 (tok, acc, rej, logp_q8)
    cands: tuple[tuple[int, int, int, int], ...]
    total: int = -1  # Σ(acc+rej)
    total_acc: int = -1  # Σacc

    def total_count(self) -> int:
        return self.total if self.total >= 0 else sum(a + r for _, a, r, _ in self.cands)

    def total_acc_count(self) -> int:
        return self.total_acc if self.total_acc >= 0 else sum(a for _, a, _, _ in self.cands)


def blend(params: BackoffParams, sources: list[Source]) -> Posterior | None:
    if not sources:
        return None

    tbl_o, tbl_s = _decay_pows(params.order_decay, params.scope_decay)
    kp = params.count_prior
    weights: list[float] = []
    for s in sources:
        c = s.total_count()
        weights.append(tbl_o[s.match_len] * tbl_s[s.scope_depth] * (c / (c + kp)))
    w_sum = sum(weights)
    if w_sum <= 0.0:
        return None

    beta_a = params.beta_a
    beta_b = params.beta_b
    # tok → [Σw·freq, Σw_hi·acc_rate, Σw_hi·p̂, Σw_hi, Σsupport]
    acc_w: dict[int, list[float]] = {}
    get_slot = acc_w.setdefault
    w_hi_sum = 0.0
    for s, w in zip(sources, weights):
        if w <= 0.0:
            continue
        hi = s.match_len >= params.p_hat_min_order  # annotation 신뢰 가능 소스인가
        w_hi = w if hi else 0.0
        w_hi_sum += w_hi
        # 소스 내부 p̂ 정규화 (top-k 잘림 보정: 소스 내 상대 질량만 신뢰)
        p_raw = [q8_to_p(q) for _, _, _, q in s.cands]
        z = sum(p_raw) or 1.0
        total_acc = s.total_acc_count()
        k_s = max(1, len(s.cands))
        freq_den = total_acc + k_s * beta_a
        for (tok, a, r, _q), pr in zip(s.cands, p_raw):
            slot = get_slot(tok, [0.0, 0.0, 0.0, 0.0, 0.0])
            slot[0] += w * ((a + beta_a) / freq_den)
            # 주의: 분모의 덧셈 그룹핑을 기존과 동일하게 유지 (float 결합 순서 = 값 보존)
            slot[1] += w_hi * ((a + beta_a) / (a + r + beta_a + beta_b))
            slot[2] += w_hi * (pr / z)
            slot[3] += w_hi
            # support = 최강 단일 소스의 관측 수. 같은 물리적 관측이 (차수×scope)개
            # 소스에 중복 계상되므로 합산하면 증거량이 ~21배 과대보고된다.
            ar = a + r
            if ar > slot[4]:
                slot[4] = ar

    if not acc_w:
        return None

    n_cand = len(acc_w)
    uni = 1.0 / n_cand
    gate = w_hi_sum / (w_hi_sum + params.anno_prior_w)  # annotation 신뢰도 ∈ (0,1)
    cands = []
    scores: dict[int, float] = {}
    for tok, (wf, wa, wph, wt, sup) in acc_w.items():
        # 게이트: 신뢰 없으면 중립값(0.5, uniform)으로 후퇴
        p_acc = 0.5 + (gate * (wa / wt - 0.5) if wt > 0 else 0.0)
        p_hat = uni + (gate * (wph / w_hi_sum - uni) if w_hi_sum > 0 else 0.0)
        freq = wf / w_sum
        # 랭킹 점수 = 빈도 × 게이트된 p̂ 보정. acc/rej 비(p_acc)는 '무엇이 다음인가'가
        # 아니라 '제안 시 붙을 확률'의 통계이므로 확장/budget 결정(proposer)에서 쓰고
        # intra-key 랭킹에는 넣지 않는다 — 다봉 문맥에서 modal 토큰을 깎는 편향 방지.
        scores[tok] = freq * (n_cand * p_hat)
        cands.append(PosteriorCand(tok=tok, p_acc=p_acc, p_hat=p_hat, support=int(sup)))
    # 상대 점수 10% 미만 차이는 동률로 보고 support(증거량)→tok로 결정한다:
    # 증거 없는 키에서 p̂ 잔차가 luck 비대칭을 만드는 것을 막고, 게이트를 통과한
    # 진짜 신호(수 배 차이)만 랭킹을 바꾼다. (결정성 I4: 전 단계 결정적)
    s_max = max(scores.values()) or 1.0
    cands.sort(key=lambda c: (-round(scores[c.tok] / s_max, 1), -c.support, c.tok))
    cands = cands[: params.max_cands]

    # p_hat 재정규화 (correction 분포로 그대로 쓰이므로 합≤1 유지)
    z = sum(c.p_hat for c in cands)
    if z > 0:
        cands = [
            PosteriorCand(c.tok, c.p_acc, c.p_hat / max(z, 1.0), c.support) for c in cands
        ]

    best_order = max(s.match_len for s in sources)
    return Posterior(cands=tuple(cands), weight=w_sum, best_order=best_order)
