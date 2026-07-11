"""proposer 4종 (a)(a′)(b)(c) — CLAUDE.md §2 sim/proposers.py.

로드맵 원본 부재로 인한 매핑 가정(docs/DECISIONS.md A-2):
  (a)  ledger   : suffix-key + outcome annotation(acc/rej/p̂/correction) — 본 연구
  (a′) dense    : 64-d dense key + positive-only value — key 효과 분리 통제군 (§7)
  (b)  positive : suffix-key + positive-only(realized count만) — value(outcome) ablation
  (c)  recycle  : Token Recycling 충실 재현 (third_party/token-recycling@1b4c05c 실측 사양)

(a)와 (b)는 같은 core.LedgerStore를 쓰고 (b)는 harvest 이벤트에서 outcome 정보만
벗겨낸다 — G2/G-R1의 "outcome annotation ablation"이 코드 분기 없이 성립한다.
추가로 vanilla(빈 draft, τ=1 sanity)를 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.signature import RollingSigStack
from core.store import LedgerStore, StoreParams
from core.types import DraftTree, Scope, Segment, VerifyOutcome


@dataclass
class ProposeCtx:
    sigs: RollingSigStack  # 현재 컨텍스트의 서명 스택 (proposer는 clone해서 사용)
    scope_stack: list[int]
    seg: Segment
    budget: int
    recent: tuple[int, ...]  # 최근 realized 토큰 (TR 루트·dense key용)
    pos: int


class BaseProposer:
    name = "base"

    def begin_request(self, scope: Scope) -> None:  # noqa: B027
        pass

    def end_request(self) -> None:  # noqa: B027
        pass

    def propose(self, ctx: ProposeCtx) -> DraftTree:
        raise NotImplementedError

    def harvest(self, ev: VerifyOutcome) -> None:  # noqa: B027
        pass

    def stats(self) -> dict:
        return {}


class VanillaProposer(BaseProposer):
    """spec off — 절대 기준선 (τ=1.0 sanity)."""

    name = "vanilla"

    def propose(self, ctx: ProposeCtx) -> DraftTree:
        return DraftTree()


class LedgerProposer(BaseProposer):
    """(a)/(b): core.LedgerStore 기반 chain(+V2 span) proposer — §3.2 규칙의 sim 대응.

    - posterior 기대 accept 확률(p_acc)로 확장, p_min 미만(reject-dominant)이면
      budget 0 + patch splice(p̂ argmax 1개 붙이고 종료)
    - V2 span은 통째 제안하되 break offset에서 budget pre-split
    - value_mode="positive"면 harvest에서 rej/topk/correction 정보를 벗겨 저장 (b)
    """

    def __init__(
        self,
        store_params: StoreParams | None = None,
        value_mode: str = "outcome",
        cum_min: float = 0.05,
        w_min: float = 1e-4,
        rej_margin: float = 0.05,
        span_min_count: int = 2,
        break_split_min: int = 2,
        name: str | None = None,
    ):
        self.store = LedgerStore(store_params or StoreParams())
        self.value_mode = value_mode
        self.cum_min = cum_min
        self.w_min = w_min
        self.rej_margin = rej_margin
        self.span_min_count = span_min_count
        self.break_split_min = break_split_min
        self.name = name or ("ledger" if value_mode == "outcome" else "positive")
        self.n_span_uses = 0
        self.n_patches = 0

    # -------------------------------------------------------------- propose
    def propose(self, ctx: ProposeCtx) -> DraftTree:
        tree = DraftTree()
        sims = ctx.sigs.clone()
        parent = -1
        budget = ctx.budget

        span = self.store.lookup_span(sims.stack_list(), ctx.scope_stack, ctx.seg)
        if span is not None and span.count >= self.span_min_count:
            self.n_span_uses += 1
            heavy = {off for off, cnt in span.breaks if cnt >= self.break_split_min}
            for off, tok in enumerate(span.tokens):
                if len(tree) >= budget:
                    break
                if off in heavy:
                    # break offset에서 budget pre-split: span 토큰과 correction(p̂ argmax)을
                    # 형제로 splice하고 이 edge 이후 확장은 중단 (§3.2). budget 엄수.
                    tree.add(tok, parent)
                    if len(tree) < budget:
                        post = self.store.lookup(sims.stack_list(), ctx.scope_stack, ctx.seg)
                        if post is not None and post.cands:
                            patch = max(post.cands, key=lambda c: (c.p_hat, -c.tok))
                            if patch.tok != tok:
                                tree.add(patch.tok, parent)
                                self.n_patches += 1
                    return tree
                parent = tree.add(tok, parent)
                sims.push(tok)

        # 기대 accept 길이 기반 확장 (§3.2): 누적 ∏p_acc가 cum_min 아래로 내려가면 중단
        cum = 1.0
        while len(tree) < budget and cum >= self.cum_min:
            post = self.store.lookup(sims.stack_list(), ctx.scope_stack, ctx.seg)
            if post is None or not post.cands or post.weight < self.w_min:
                break
            top = post.cands[0]
            # rej-dominant: '신뢰 게이트를 통과한' 기각 우세만 인정한다 (p_acc는 λ-질량
            # 게이트가 적용된 값 — 저특이 키의 기각 노이즈와, 진성 다봉 분포(modal이라도
            # p_acc<0.5)를 margin으로 걸러낸다). 다봉 위치에서 modal 제안은 여전히 +EV다.
            rej_dominant = top.support > 0 and top.p_acc < 0.5 - self.rej_margin
            if rej_dominant:
                # reject-dominant edge(§3.2): 그 edge의 후속엔 budget 0 (leaf로만 남김),
                # correction(p̂ argmax)을 splice해 체인은 patch를 통해 계속 이어간다 —
                # 교정 후 미래는 다시 예측 가능하다는 OSD류 관찰의 구현. budget 엄수:
                # 남은 예산이 1이면 leaf 없이 patch(교정 기대값이 더 큼)만 넣는다.
                patch = max(post.cands, key=lambda c: (c.p_hat, -c.tok))
                if patch.tok != top.tok:
                    if budget - len(tree) >= 2:
                        tree.add(top.tok, parent)  # 수락 가능성 보존용 leaf (확장 없음)
                    parent = tree.add(patch.tok, parent)
                    self.n_patches += 1
                    sims.push(patch.tok)
                    cum *= max(patch.p_acc, 0.05)
                    continue
                # patch == top: 교정 분포도 같은 토큰 — leaf로만 남기고 중단 (budget 0)
                tree.add(top.tok, parent)
                break
            parent = tree.add(top.tok, parent)
            sims.push(top.tok)
            cum *= max(top.p_acc, 0.05)
        return tree

    # -------------------------------------------------------------- harvest
    def harvest(self, ev: VerifyOutcome) -> None:
        if self.value_mode == "positive":
            realized = ev.realized()
            n = len(realized)
            ev = VerifyOutcome(
                scope=ev.scope,
                ctx_tail=ev.ctx_tail,
                draft_ids=tuple(realized[:-1]),
                accepted_len=n - 1,
                bonus_id=realized[-1],
                topk_ids=tuple(() for _ in range(n)),
                topk_logp_q8=tuple(() for _ in range(n)),
                seg=ev.seg,
                file_id=ev.file_id,
                t_us=ev.t_us,
            )
        self.store.harvest([ev])
        self.store.drain()  # sim은 step 동기 drain (결정성 I4)

    def end_request(self) -> None:
        self.store.flush_runs()

    def stats(self) -> dict:
        d = self.store.stats().as_dict()
        d["span_uses"] = self.n_span_uses
        d["patches"] = self.n_patches
        return d


# --- Token Recycling 정적 트리 템플릿 "2.2.2" (원 구현 tree_template_.py:3-10 그대로) ---
TR_TREE_TEMPLATE: list[list[int]] = [
    [0], [1], [2], [3], [4], [5], [6], [7],
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
    [1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1],
    [4, 0], [5, 0], [6, 0], [7, 0],
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7],
    [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 3, 0], [0, 4, 0], [0, 5, 0],
    [0, 6, 0], [0, 7, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
    [5, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4], [0, 0, 1, 0],
    [0, 0, 1, 1], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0], [0, 1, 0, 0], [0, 2, 0, 0],
    [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0],
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0],
    [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0],
]

TR_K = 8  # 원 구현에서 top-k가 리터럴 8로 고정 (inference_recycling.py:131)


def _tr_tree_wiring(template: list[list[int]]) -> list[tuple[int, int]]:
    """[(parent_node_idx, child_slot)] — template 순서 그대로."""
    index_of: dict[tuple[int, ...], int] = {}
    wiring = []
    for i, path in enumerate(template):
        parent = index_of[tuple(path[:-1])] if len(path) > 1 else -1
        wiring.append((parent, path[-1]))
        index_of[tuple(path)] = i
    return wiring


class TokenRecyclingProposer(BaseProposer):
    """(c) Token Recycling sim 재현 (arXiv 2408.08696, 코드 실측 사양).

    - M: tok → 최근 top-8 next-token id (zeros init, overwrite/last-write-wins)
    - 정적 트리 80노드(depth 6), BFS로 M에서 채움; 루트 = 직전 realized 토큰
    - 원 구현은 트리 '모든' 노드 위치의 logits로 M을 갱신하지만, trace에는 realized
      경로의 top-k만 있으므로 realized 경로만 갱신한다(과소평가 방향 — DECISIONS A-2)
    - M은 원 구현처럼 요청 경계를 넘어 영속하되, tenant 격리(§10)를 위해 per-tenant
    """

    name = "recycle"

    def __init__(self) -> None:
        self._m_by_tenant: dict[str, dict[int, list[int]]] = {}
        self._m: dict[int, list[int]] = {}
        self._wiring = _tr_tree_wiring(TR_TREE_TEMPLATE)
        self._zero_row = [0] * TR_K

    def begin_request(self, scope: Scope) -> None:
        self._m = self._m_by_tenant.setdefault(scope.tenant, {})

    def propose(self, ctx: ProposeCtx) -> DraftTree:
        if not ctx.recent:
            return DraftTree()
        root = ctx.recent[-1]
        tree = DraftTree()
        toks = [0] * len(self._wiring)
        for i, (parent, slot) in enumerate(self._wiring):
            parent_tok = root if parent == -1 else toks[parent]
            row = self._m.get(parent_tok, self._zero_row)
            toks[i] = row[slot]
            tree.add(toks[i], parent)
        return tree

    def harvest(self, ev: VerifyOutcome) -> None:
        realized = ev.realized()
        prev = ev.ctx_tail[-1] if ev.ctx_tail else None
        for pos, _tok in enumerate(realized):
            if prev is not None and pos < len(ev.topk_ids) and ev.topk_ids[pos]:
                row = list(ev.topk_ids[pos][:TR_K])
                if len(row) < TR_K:
                    row += [0] * (TR_K - len(row))
                self._m[prev] = row  # overwrite — 원 구현 :132와 동일
            prev = realized[pos]

    def stats(self) -> dict:
        rows = sum(len(m) for m in self._m_by_tenant.values())
        return {"rows": rows, "bytes": rows * TR_K * 4, "hit_rate": 0.0}


def make_proposer(spec: dict) -> BaseProposer:
    """config dict → proposer.

    kind ∈ {ledger(a), positive(b), dense(a′), recycle(c), vanilla}.
    """
    kind = spec["kind"]
    if kind == "vanilla":
        return VanillaProposer()
    if kind == "recycle":
        return TokenRecyclingProposer()
    if kind in ("ledger", "positive"):
        sp = StoreParams.from_dict(spec.get("store"))
        return LedgerProposer(
            store_params=sp,
            value_mode="outcome" if kind == "ledger" else "positive",
            cum_min=spec.get("cum_min", 0.05),
            w_min=spec.get("w_min", 1e-4),
            span_min_count=spec.get("span_min_count", 2),
            break_split_min=spec.get("break_split_min", 2),
            name=spec.get("name"),
        )
    if kind == "dense":
        from sim.dense_key import DenseKeyProposer

        return DenseKeyProposer(
            dims=spec.get("dims", 64),
            window=spec.get("window", 16),
            max_dist=spec.get("max_dist", 0.35),
            use_ivfpq=spec.get("ivfpq", False),
            name=spec.get("name", "dense"),
        )
    raise ValueError(f"unknown proposer kind: {kind}")


@dataclass
class ProposerSpec:
    kind: str
    params: dict = field(default_factory=dict)
