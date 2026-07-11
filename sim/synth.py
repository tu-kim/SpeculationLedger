"""합성 trace 생성기 — Phase 0 하네스 검증용 (GPU 부재 환경).

실제 trace(OpenCode×SWE-bench×vLLM tracer)가 확보되기 전까지 sim 파이프라인의
기능·결정성(I4)·gate 산식을 검증하기 위한 대체물이다. 이것으로 산출된 gates.json은
trace_provenance="synthetic"으로 표기되며 **연구 가설의 판정 근거가 아니다**
(docs/DECISIONS.md A-3).

생성 모델 (agent turn 구조를 흉내):
  turn := think(주제 어휘 재조합) → tool(JSON 템플릿, 고재발)
          [edit이면 tool 인자 안 code 구간: 파일 라인 재방출 + 점변이(=span break 구조)]
          → text(요약)
  - 같은 repo의 세션들은 파일 내용·토픽을 공유 → repo-scope 재발
  - edit는 파일 상태를 실제로 변이시키고 events(file_edit)를 기록 → epoch 검증 경로
  - topk는 true token 고확률 + 시드된 distractor로 목표분포 p̂를 흉내낸다
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

from core.types import Segment, logp_to_q8, stable_u64

_SPECIAL = 16  # 0..15 예약


@dataclass(frozen=True)
class SynthParams:
    seed: int = 20260711
    vocab: int = 8000
    tenants: int = 1
    repos: int = 2
    sessions_per_repo: int = 4
    turns_per_session: int = 6
    files_per_repo: int = 3
    file_lines: int = 10
    line_len: int = 8
    think_sents: tuple[int, int] = (3, 7)
    sent_len: tuple[int, int] = (5, 10)
    think_novelty: float = 0.18
    think_templates: int = 14  # repo별 정형 reasoning 문장 템플릿 수 (실측: 매우 정형적)
    file_locality: float = 0.6  # 직전 턴과 같은 파일을 계속 작업할 확률
    edit_prob: float = 0.55
    edit_lines: tuple[int, int] = (3, 6)
    mutate_toks: tuple[int, int] = (1, 2)
    consistent_mut: float = 0.7  # 변이 중 '일관 rename'(X→Y 반복) 비율 — correction 재발원
    k_top: int = 8

    @staticmethod
    def from_dict(d: dict | None) -> "SynthParams":
        if not d:
            return SynthParams()
        d = dict(d)
        for k in ("think_sents", "sent_len", "edit_lines", "mutate_toks"):
            if k in d:
                d[k] = tuple(d[k])
        return SynthParams(**d)


def _tok(word: str, vocab: int) -> int:
    return _SPECIAL + stable_u64("w", word) % (vocab - _SPECIAL)


class _Vocab:
    def __init__(self, p: SynthParams):
        self.p = p
        self.json_struct = [_tok(w, p.vocab) for w in
                            ["{", "}", "[", "]", ":", ",", '"', "name", "arguments", "path",
                             "read_file", "edit_file", "content", "old", "new"]]
        # 전역 공용 어휘 (모든 repo가 공유 → global-scope 재발원)
        self.common = [_tok(f"common{i}", p.vocab) for i in range(200)]

    def topic_bank(self, repo: str) -> list[int]:
        return [_tok(f"{repo}:topic{i}", self.p.vocab) for i in range(120)]

    def think_templates(self, repo: str) -> list[list[int]]:
        """정형 reasoning 문장 템플릿: 고정 토큰열 + 슬롯(-1) 1~2개."""
        rng = random.Random(stable_u64("tmpl", repo))
        bank = self.topic_bank(repo)
        out = []
        for ti in range(self.p.think_templates):
            ln = rng.randint(*self.p.sent_len)
            t = [bank[rng.randrange(len(bank))] if rng.random() < 0.5
                 else self.common[rng.randrange(len(self.common))]
                 for _ in range(ln)]
            for _ in range(rng.randint(1, 2)):
                t[rng.randrange(ln)] = -1  # slot
            out.append(t)
        return out

    def code_word(self, repo: str, i: int) -> int:
        return _tok(f"{repo}:code{i}", self.p.vocab)


def _emit_topk(rng: random.Random, true_tok: int, alts: list[int], vocab: int, k: int):
    """true token + 문맥-그럴듯한 대안(alts) + 랜덤 잔여로 구성한 top-k (ids, q8).

    실제 target 분포의 핵심 성질을 흉내낸다: 분기점(툴 선택, 변이 지점)에서는
    실제 대안들에 질량이 실린다 — p̂ 기반 correction/감쇠 로직이 학습할 대상.
    """
    true_lp = min(-0.01, rng.gauss(-0.18, 0.12))
    ids = [true_tok]
    lps = [true_lp]
    lp = true_lp - rng.uniform(0.5, 1.0)
    for a in alts:
        if len(ids) >= k:
            break
        if a != true_tok and a not in ids:
            ids.append(a)
            lps.append(lp)
            lp -= rng.uniform(0.2, 0.8)
    while len(ids) < k:
        lp -= rng.uniform(0.4, 1.4)
        d = _SPECIAL + rng.randrange(vocab - _SPECIAL)
        if d in ids:
            d = _SPECIAL + (d + 1 - _SPECIAL) % (vocab - _SPECIAL)
        ids.append(d)
        lps.append(lp)
    return ids, [logp_to_q8(x) for x in lps]


@dataclass
class _Req:
    tokens: list[int] = field(default_factory=list)
    seg: list[int] = field(default_factory=list)
    alts: list[list[int]] = field(default_factory=list)  # 위치별 그럴듯한 대안 (topk 재료)
    events: list[dict] = field(default_factory=list)

    def put(self, toks: list[int], seg: Segment, alts: list[list[int]] | None = None) -> None:
        self.tokens.extend(toks)
        self.seg.extend([int(seg)] * len(toks))
        if alts is None:
            self.alts.extend([[] for _ in toks])
        else:
            assert len(alts) == len(toks)
            self.alts.extend(alts)


class _RepoState:
    def __init__(self, v: _Vocab, p: SynthParams, tenant: str, repo: str):
        rng = random.Random(stable_u64("repo-init", tenant, repo))
        self.repo = repo
        self.vocab = p.vocab
        self.files: dict[str, list[list[int]]] = {}
        for fi in range(p.files_per_repo):
            path = f"src/{repo}_mod{fi}.py"
            lines = []
            for li in range(p.file_lines):
                line = [v.code_word(repo, rng.randrange(400)) for _ in range(p.line_len)]
                lines.append(line)
            self.files[path] = lines

    def rename_of(self, tok: int) -> int:
        """tok의 repo-일관 rename 타깃 (결정적)."""
        return _tok(f"{self.repo}:ren:{tok}", self.vocab)

    def mutate_tok(self, tok: int, rng: random.Random, p: SynthParams) -> int:
        """일관 rename(같은 X는 항상 같은 Y로) 또는 신규 랜덤 변이."""
        if rng.random() < p.consistent_mut:
            return self.rename_of(tok)
        return _tok(f"mut{rng.randrange(4000)}", self.vocab)


def _gen_think(
    rng: random.Random,
    v: _Vocab,
    bank: list[int],
    templates: list[list[int]],
    p: SynthParams,
    r: _Req,
) -> None:
    """정형 템플릿 문장(슬롯 치환) 위주 + 가끔 완전 신규 문장 — 실제 reasoning의 정형성."""
    n_sent = rng.randint(*p.think_sents)
    for _ in range(n_sent):
        if rng.random() < p.think_novelty:
            ln = rng.randint(*p.sent_len)
            sent = [_SPECIAL + rng.randrange(p.vocab - _SPECIAL) if rng.random() < 0.5
                    else bank[rng.randrange(len(bank))] for _ in range(ln)]
            alts = [[bank[rng.randrange(len(bank))]] for _ in sent]
            r.put(sent, Segment.THINK, alts)
            continue
        t = templates[rng.randrange(len(templates))]
        sent, alts = [], []
        for tok in t:
            if tok == -1:
                sent.append(bank[rng.randrange(len(bank))])
                alts.append([bank[rng.randrange(len(bank))], bank[rng.randrange(len(bank))]])
            else:
                sent.append(tok)
                alts.append([v.common[rng.randrange(60)]])
        r.put(sent, Segment.THINK, alts)


def _tool_alts(
    v: _Vocab, toks: list[int], path_tok: int, other_tool: int, sibling_paths: list[int]
) -> list[list[int]]:
    """툴 JSON 위치별 대안: 툴 이름 위치엔 다른 툴, 경로 위치엔 형제 경로들(다봉 분포)."""
    js = set(v.json_struct)
    out = []
    for t in toks:
        if t in (v.json_struct[10], v.json_struct[11]):
            out.append([other_tool])
        elif t == path_tok:
            out.append([s for s in sibling_paths if s != path_tok])
        elif t in js:
            out.append([v.json_struct[5], v.json_struct[1]])  # 구조 토큰 혼동쌍
        else:
            out.append([])
    return out


def _gen_tool_read(
    v: _Vocab, path: str, sibling_paths: list[int], p: SynthParams, r: _Req
) -> None:
    js = v.json_struct
    q = js[6]
    call = [js[0], q, js[7], q, js[4], q, js[10], q, js[3 + 0]]
    call += [js[0], q, js[9], q, js[4], q, _tok(path, p.vocab), q, js[1], js[1]]
    r.put(call, Segment.TOOL, _tool_alts(v, call, _tok(path, p.vocab), js[11], sibling_paths))


def _gen_tool_edit(
    rng: random.Random,
    v: _Vocab,
    state: _RepoState,
    path: str,
    sibling_paths: list[int],
    p: SynthParams,
    r: _Req,
) -> None:
    """str_replace형 편집: old 블록(현재 내용 '정확' 재방출) + new 블록(변이 적용).

    - old 블록: 이전 편집의 new 내용이 이후 old로 재등장 → V2 span 재발의 원천
    - new 블록: 점변이. consistent_mut 비율은 repo-일관 rename(X→Y 반복)
      → 기각→correction 패턴이 재발해 outcome annotation이 학습할 대상이 된다
    """
    js = v.json_struct
    q = js[6]
    head = [js[0], q, js[7], q, js[4], q, js[11], q, js[2 + 1],
            js[0], q, js[9], q, js[4], q, _tok(path, p.vocab), q, js[4 - 3],
            q, js[13], q, js[4], q]
    r.put(head, Segment.TOOL, _tool_alts(v, head, _tok(path, p.vocab), js[10], sibling_paths))

    lines = state.files[path]
    n_edit = min(rng.randint(*p.edit_lines), len(lines))
    # 핫 리전 편향: 낮은 라인이 반복 편집될 확률↑ → 같은 old 블록 재발
    start = int((len(lines) - n_edit + 1) * (rng.random() ** 2))
    old_body: list[int] = []
    old_alts: list[list[int]] = []
    new_lines: list[list[int]] = []
    new_alts_lines: list[list[list[int]]] = []
    for li in range(start, start + n_edit):
        old_body.extend(lines[li])
        line = list(lines[li])
        a_line: list[list[int]] = [[] for _ in line]
        for _ in range(rng.randint(*p.mutate_toks)):
            j = rng.randrange(len(line))
            old_tok = line[j]
            line[j] = state.mutate_tok(old_tok, rng, p)
            a_line[j] = [old_tok]  # 변이 지점의 대안 = 기존 토큰 X
        new_lines.append(line)
        new_alts_lines.append(a_line)
    # old 블록의 대안: 각 토큰의 일관 rename 타깃 (모델이 '편집 반영 여부'를 헷갈리는 상황)
    old_alts = [[state.rename_of(t)] for t in old_body]
    r.put(old_body, Segment.CODE, old_alts)  # old 블록

    mid = [q, js[4 - 3], q, js[14], q, js[4], q]
    r.put(mid, Segment.TOOL)
    bump_pos = len(r.tokens) - 1  # old→new 경계: 여기서 epoch bump (A-7)

    new_body: list[int] = []
    new_alts: list[list[int]] = []
    for off, li in enumerate(range(start, start + n_edit)):
        lines[li] = new_lines[off]  # 파일 상태 갱신
        new_body.extend(new_lines[off])
        new_alts.extend(new_alts_lines[off])
    r.put(new_body, Segment.CODE, new_alts)  # new 블록

    tail = [q, js[1], js[1]]
    r.put(tail, Segment.TOOL)
    # invalidation event: pos = 생성 완료(파일 결속 범위 끝), bump_pos = new 인자 시작 경계
    r.events.append(
        {"pos": len(r.tokens) - 1, "bump_pos": bump_pos, "type": "file_edit", "file": path}
    )


def _gen_text(rng: random.Random, v: _Vocab, bank: list[int], p: SynthParams, r: _Req) -> None:
    ln = rng.randint(*p.sent_len)
    sent = [v.common[rng.randrange(60)] if rng.random() < 0.6 else bank[rng.randrange(len(bank))]
            for _ in range(ln)]
    r.put(sent, Segment.TEXT, [[v.common[rng.randrange(60)]] for _ in sent])


def generate(params: SynthParams, out_path: str) -> dict:
    p = params
    v = _Vocab(p)
    tokenizer_hash = hashlib.blake2b(
        json.dumps({"vocab": p.vocab, "toy": 1}, sort_keys=True).encode(), digest_size=8
    ).hexdigest()

    # (tenant, repo, session, turn) 순회를 세션 간 round-robin으로 섞어 ts 부여
    turn_specs = []
    for ti in range(p.tenants):
        tenant = f"tenant{ti}"
        for ri in range(p.repos):
            repo = f"{tenant}/repo{ri}"
            for si in range(p.sessions_per_repo):
                session = f"{repo}/s{si}"
                for tu in range(p.turns_per_session):
                    turn_specs.append((tenant, repo, session, tu))
    turn_specs.sort(key=lambda x: (x[3], x[0], x[1], x[2]))  # turn index 우선 → 세션 인터리브

    states: dict[str, _RepoState] = {}
    last_path: dict[str, str] = {}  # session → 직전 작업 파일 (locality)
    n_records = 0
    total_tokens = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ts, (tenant, repo, session, turn) in enumerate(turn_specs):
            if repo not in states:
                states[repo] = _RepoState(v, p, tenant, repo)
            state = states[repo]
            rng = random.Random(stable_u64("turn", str(p.seed), session, str(turn)))
            bank = v.topic_bank(repo)
            templates = v.think_templates(repo)
            r = _Req()

            _gen_think(rng, v, bank, templates, p, r)
            paths = sorted(state.files)
            prev = last_path.get(session)
            if prev is not None and rng.random() < p.file_locality:
                path = prev  # 파일 locality: 같은 파일을 이어서 작업
            else:
                path = paths[rng.randrange(len(paths))]
            last_path[session] = path
            sibling_toks = [_tok(pt, p.vocab) for pt in paths]
            if rng.random() < p.edit_prob:
                _gen_tool_edit(rng, v, state, path, sibling_toks, p, r)
            else:
                _gen_tool_read(v, path, sibling_toks, p, r)
            _gen_text(rng, v, bank, p, r)

            steps = []
            tki_all, tkq_all = [], []
            for i, tok in enumerate(r.tokens):
                ids, q8 = _emit_topk(rng, tok, r.alts[i], p.vocab, p.k_top)
                tki_all.append(ids)
                tkq_all.append(q8)
                steps.append(
                    {
                        "pos": i,
                        "proposed": [],
                        "accepted_len": 0,
                        "bonus": int(tok),
                        "topk_ids": [ids],
                        "topk_logp_q8": [q8],
                        "seg": [r.seg[i]],
                        "t_us": 0,
                    }
                )
            sha = hashlib.blake2b(
                ",".join(map(str, r.tokens)).encode(), digest_size=16
            ).hexdigest()
            rec = {
                "schema_version": 1,
                "request_id": f"{session}/t{turn}",
                "ts": ts,
                "scope": {
                    "tenant": tenant,
                    "repo": repo,
                    "session": session,
                    "instance_id": f"synth-{session}",
                },
                "model": "synthetic-oracle",
                "tokenizer_hash": tokenizer_hash,
                "steps": steps,
                "final_text_sha": sha,
            }
            if r.events:
                rec["events"] = r.events
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
            n_records += 1
            total_tokens += len(r.tokens)

    meta = {
        "params": asdict(p),
        "tokenizer_hash": tokenizer_hash,
        "records": n_records,
        "tokens": total_tokens,
        "provenance": "synthetic",
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, sort_keys=True, indent=1)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="합성 trace 생성 (schema v1 JSONL)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="synth params yaml (선택)")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    d = {}
    if args.config:
        import yaml

        with open(args.config, encoding="utf-8") as f:
            d = (yaml.safe_load(f) or {}).get("synth", {})
    if args.seed is not None:
        d["seed"] = args.seed
    meta = generate(SynthParams.from_dict(d), args.out)
    print(json.dumps({k: meta[k] for k in ("records", "tokens", "tokenizer_hash")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
