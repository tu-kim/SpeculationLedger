"""oracle replay: trace → proposer별 oracle τ, 재발률, hit-rate (CLAUDE.md §3.5).

oracle 검증 = greedy strict 검증의 상한과 동치: draft tree에서 truth 스트림과 일치하는
최장 루트 경로를 수락하고, 다음 truth 토큰을 bonus로 붙인다. τ_step = accepted+1.
verify 직후 계약 §3.3 형태의 VerifyOutcome을 만들어 proposer에 되먹인다(학습 경로).

trace의 topk는 realized 경로 위치에만 존재한다 — counterfactual(기각 가지) 위치의
target 분포는 오프라인에서 재구성 불가하며, 모든 proposer에 동일 제약으로 적용된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.signature import MAX_ORDER, RollingSigStack, SuffixAutomaton
from core.types import DraftTree, InvalidationEvent, Segment, VerifyOutcome
from sim.convert import TraceRequest
from sim.proposers import BaseProposer, LedgerProposer, ProposeCtx

_SEG_NAMES = {0: "think", 1: "tool", 2: "code", 3: "text"}
_RECENT_WINDOW = 32


def oracle_accept(
    tree: DraftTree, truth: list[int], p: int
) -> tuple[int, list[int], int | None]:
    """트리에서 truth[p:]와 일치하는 최장 경로.

    반환: (accepted_len, 경로 토큰들, 기각 후보 tok | None).
    기각 후보 = 마지막 수락 노드의 첫 자식(삽입 순서) — greedy 검증기가 그 위치에서
    비교했을 draft 토큰. 자식이 없으면(트리 소진) None.
    """
    children: list[list[int]] = [[] for _ in range(len(tree.nodes) + 1)]
    for i, n in enumerate(tree.nodes):
        children[n.parent + 1].append(i)

    path: list[int] = []
    cur = 0  # children 인덱스 공간에서 루트 = 0 (parent -1)
    while True:
        kids = children[cur]
        if not kids:
            return len(path), path, None
        depth = len(path)
        hit = None
        for k in kids:
            if p + depth < len(truth) and tree.nodes[k].tok == truth[p + depth]:
                hit = k
                break
        if hit is None:
            return len(path), path, tree.nodes[kids[0]].tok
        path.append(tree.nodes[hit].tok)
        cur = hit + 1


@dataclass
class RequestMetrics:
    request_id: str
    ts: int
    session: str
    steps: int = 0
    tokens: int = 0
    drafted_tokens: int = 0
    drafted_steps: int = 0
    seg_steps: dict[str, int] = field(default_factory=dict)
    seg_tokens: dict[str, int] = field(default_factory=dict)

    @property
    def tau(self) -> float:
        return self.tokens / self.steps if self.steps else 0.0

    def as_row(self, proposer: str) -> dict:
        row = {
            "proposer": proposer,
            "request_id": self.request_id,
            "ts": self.ts,
            "steps": self.steps,
            "tokens": self.tokens,
            "tau": round(self.tau, 6),
            "drafted_tokens": self.drafted_tokens,
            "drafted_steps": self.drafted_steps,
        }
        for s in _SEG_NAMES.values():
            st = self.seg_steps.get(s, 0)
            tk = self.seg_tokens.get(s, 0)
            row[f"steps_{s}"] = st
            row[f"tau_{s}"] = round(tk / st, 6) if st else 0.0
        return row


@dataclass
class ReplayResult:
    proposer: str
    requests: list[RequestMetrics]
    proposer_stats: dict

    def totals(self) -> dict:
        steps = sum(r.steps for r in self.requests)
        tokens = sum(r.tokens for r in self.requests)
        seg_steps: dict[str, int] = {}
        seg_tokens: dict[str, int] = {}
        for r in self.requests:
            for s, v in r.seg_steps.items():
                seg_steps[s] = seg_steps.get(s, 0) + v
            for s, v in r.seg_tokens.items():
                seg_tokens[s] = seg_tokens.get(s, 0) + v
        out = {
            "steps": steps,
            "tokens": tokens,
            "tau": round(tokens / steps, 6) if steps else 0.0,
            "drafted_tokens_per_step": round(
                sum(r.drafted_tokens for r in self.requests) / steps, 6
            )
            if steps
            else 0.0,
            "per_seg_tau": {
                s: round(seg_tokens.get(s, 0) / seg_steps[s], 6)
                for s in sorted(seg_steps)
                if seg_steps[s]
            },
            "per_seg_steps": {s: seg_steps[s] for s in sorted(seg_steps)},
        }
        return out

    def learning_curve(self) -> list[float]:
        return [round(r.tau, 6) for r in self.requests]


def run_replay(reqs: list[TraceRequest], proposer: BaseProposer, budget: int) -> ReplayResult:
    out: list[RequestMetrics] = []
    for req in reqs:
        scope_stack = [sid for _, sid in req.scope.scope_stack()]
        sigs = RollingSigStack()
        recent: list[int] = []
        n = len(req.tokens)
        rm = RequestMetrics(request_id=req.request_id, ts=req.ts, session=req.scope.session)
        proposer.begin_request(req.scope)

        # CODE 위치 → epoch domain(file_id) 매핑 (events 기반)
        file_ids = [0] * n
        prev_end = 0
        for ev in req.events:
            if ev.type == "file_edit":
                fid = InvalidationEvent(req.scope, ev.file).file_id()
                for i in range(prev_end, min(ev.pos + 1, n)):
                    file_ids[i] = fid
                prev_end = ev.pos + 1
        ev_idx = 0

        p = 0
        while p < n:
            seg = Segment(req.seg[p])
            ctx = ProposeCtx(
                sigs=sigs,
                scope_stack=scope_stack,
                seg=seg,
                budget=budget,
                recent=tuple(recent[-_RECENT_WINDOW:]),
                pos=p,
            )
            tree = proposer.propose(ctx)
            acc, path, rej_tok = oracle_accept(tree, req.tokens, p)
            max_take = n - p - 1  # bonus가 존재해야 함
            if acc > max_take:
                acc, rej_tok = max_take, None  # 시퀀스 끝 절단 — 가짜 rejection 금지
            bonus = req.tokens[p + acc]

            draft_ids = tuple(path[:acc]) + ((rej_tok,) if rej_tok is not None else ())
            n_real = acc + 1
            seg_arr = tuple(req.seg[p : p + n_real])
            fid = next((file_ids[i] for i in range(p, p + n_real) if file_ids[i]), 0)
            outcome = VerifyOutcome(
                scope=req.scope,
                ctx_tail=tuple(req.tokens[max(0, p - MAX_ORDER) : p]),
                draft_ids=draft_ids,
                accepted_len=acc,
                bonus_id=bonus,
                topk_ids=tuple(req.topk_ids[p : p + n_real]),
                topk_logp_q8=tuple(req.topk_logp_q8[p : p + n_real]),
                seg=seg_arr,
                file_id=fid,
            )
            proposer.harvest(outcome)

            # 이 step 구간에서 도달한 invalidation 이벤트 발화 (§3.4, bump_pos 기준 A-7)
            while ev_idx < len(req.events) and req.events[ev_idx].effective_bump_pos() < p + n_real:
                e = req.events[ev_idx]
                if e.type == "file_edit" and isinstance(proposer, LedgerProposer):
                    inv = InvalidationEvent(req.scope, e.file)
                    proposer.store.bump_epoch(req.scope.session_id(), inv.file_id())
                    proposer.store.bump_epoch(req.scope.repo_id(), inv.file_id())
                ev_idx += 1

            seg_name = _SEG_NAMES.get(int(seg), "text")
            rm.steps += 1
            rm.tokens += n_real
            rm.seg_steps[seg_name] = rm.seg_steps.get(seg_name, 0) + 1
            rm.seg_tokens[seg_name] = rm.seg_tokens.get(seg_name, 0) + n_real
            rm.drafted_tokens += len(tree)
            rm.drafted_steps += 1 if len(tree) else 0

            for i in range(p, p + n_real):
                sigs.push(req.tokens[i])
                recent.append(req.tokens[i])
            del recent[:-_RECENT_WINDOW]
            p += n_real

        proposer.end_request()
        out.append(rm)
    return ReplayResult(proposer=proposer.name, requests=out, proposer_stats=proposer.stats())


# ------------------------------------------------------------------ G1 재발률
def recurrence_stats(reqs: list[TraceRequest]) -> dict:
    """suffix automaton 기반 정확 재발 통계 (proposer 비의존).

    scope_mode별로 스트림을 이어 붙이며(요청 ts 순), 각 위치에서 '직전 컨텍스트+이 토큰'
    suffix가 과거에 등장했던 최장 길이를 잰다. rate_ge[o] = P(match_len ≥ o).
    """
    out = {}
    for mode in ("session", "repo"):
        sams: dict[str, SuffixAutomaton] = {}
        total = 0
        ge = {o: 0 for o in range(1, MAX_ORDER + 1)}
        hist: dict[int, int] = {}
        for req in reqs:
            key = req.scope.session if mode == "session" else req.scope.repo
            sam = sams.setdefault(key, SuffixAutomaton())
            for tok in req.tokens:
                m = sam.extend(tok)
                total += 1
                b = min(m, 32)
                hist[b] = hist.get(b, 0) + 1
                for o in ge:
                    if m >= o:
                        ge[o] += 1
        out[mode] = {
            "positions": total,
            "rate_ge": {str(o): round(ge[o] / total, 6) if total else 0.0 for o in ge},
            "match_len_hist": {str(k): hist[k] for k in sorted(hist)},
        }
    return out
