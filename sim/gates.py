"""gate 판정 러너: config yaml → replay → results/<exp_id>/gates.json (CLAUDE.md §8, §4).

- config 하나 = 실험 하나. 결과 디렉토리에 config 사본·git hash·gates.json·metrics.parquet 저장.
- gates.json은 byte-결정적이다(I4): 정렬 키, float 6자리 반올림, 타임스탬프 없음.
- 로드맵 원본 부재로 G1/G2/G3/G-R1의 운영적 판정식·임계값은 **가정**이며 config의
  thresholds 블록과 docs/DECISIONS.md A-1에 기록된다. 합성 trace 결과는
  trace_provenance="synthetic"으로 연구 판정에 쓰지 않는다(A-3).

판정식 (A-1):
  G1   재발률: within-session P(suffix match_len ≥ 4) ≥ g1_min_rate_order4_session
  G2   oracle τ: role=a proposer의 전체 τ ≥ g2_min_tau_ledger
  G3   hit-rate–size: bytes ≤ g3_budget_bytes인 cap 중 hit_rate ≥
       g3_min_hit_frac × (무제한 hit_rate)가 존재
  G-R1 outcome annotation 전제: (τ_a − τ_b)/τ_b ≥ gr1_min_rel_gain (전체) 또는
       support ≥ gr1_min_seg_steps인 seg 중 상대 이득 최대값이 임계 이상
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess

import yaml

from core.store import StoreParams
from sim.convert import read_traces
from sim.proposers import make_proposer
from sim.replay import recurrence_stats, run_replay
from sim.synth import SynthParams, generate

_DEFAULT_THRESHOLDS = {
    "assumed": True,
    "g1_min_rate_order4_session": 0.30,
    "g2_min_tau_ledger": 2.0,
    "g3_budget_bytes": 64 * 1024 * 1024,
    "g3_min_hit_frac": 0.8,
    "gr1_min_rel_gain": 0.10,
    "gr1_min_seg_steps": 300,
}


def _round_floats(obj, nd: int = 6):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, nd) for v in obj]
    return obj


def canonical_json(doc: dict) -> str:
    return json.dumps(_round_floats(doc), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _git_hash() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return "nogit"


def _resolve_traces(cfg: dict) -> tuple[list[str], dict]:
    tr = cfg["trace"]
    paths = tr.get("paths", [])
    import glob as _g

    have = [p for pat in paths for p in _g.glob(pat)]
    meta = {}
    if not have:
        synth = tr.get("synth")
        if not synth:
            raise FileNotFoundError(f"traces not found and no synth params: {paths}")
        out = paths[0] if paths else "traces/synth/auto.jsonl"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        meta = generate(SynthParams.from_dict(synth), out)
        have = [out]
        paths = [out]
    else:
        meta_path = have[0] + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
    return paths, meta


def _run_size_sweep(cfg: dict, reqs, budget: int) -> list[dict]:
    sw = cfg.get("size_sweep")
    if not sw:
        return []
    rows = []
    for cap in sw["max_entries"]:
        spec = json.loads(json.dumps(sw["proposer"]))  # deep copy
        store = spec.setdefault("store", {})
        store["max_entries"] = cap
        spec.setdefault("name", f"sweep_{cap}")
        prop = make_proposer(spec)
        res = run_replay(reqs, prop, budget)
        st = prop.stats()
        rows.append(
            {
                "max_entries": cap,
                "bytes": st.get("bytes", 0),
                "hit_rate": st.get("hit_rate", 0.0),
                "tau": res.totals()["tau"],
                "evictions": st.get("evictions", 0),
            }
        )
    return rows


def _judge(cfg, thresholds, recur, by_role, sweep) -> dict:
    gates = {}
    want = cfg.get("gates", ["G1", "G2", "G3", "G-R1"])

    if "G1" in want:
        v = recur["session"]["rate_ge"]["4"]
        thr = thresholds["g1_min_rate_order4_session"]
        gates["G1"] = {
            "value": v,
            "threshold": thr,
            "pass": bool(v >= thr),
            "metric": "within-session P(match_len>=4)",
        }

    if "G2" in want and "a" in by_role:
        v = by_role["a"]["totals"]["tau"]
        thr = thresholds["g2_min_tau_ledger"]
        others = {
            r: by_role[r]["totals"]["tau"] for r in sorted(by_role) if r != "a"
        }
        gates["G2"] = {
            "value": v,
            "threshold": thr,
            "pass": bool(v >= thr),
            "metric": "oracle tau of (a) ledger",
            "baselines_tau": others,
        }

    if "G3" in want and sweep:
        unbounded = next((r for r in sweep if r["max_entries"] == 0), None)
        thr_frac = thresholds["g3_min_hit_frac"]
        budget_b = thresholds["g3_budget_bytes"]
        best = 0.0
        ok = False
        if unbounded and unbounded["hit_rate"] > 0:
            for r in sweep:
                if r["max_entries"] != 0 and r["bytes"] <= budget_b:
                    frac = r["hit_rate"] / unbounded["hit_rate"]
                    best = max(best, frac)
                    if frac >= thr_frac:
                        ok = True
        gates["G3"] = {
            "value": best,
            "threshold": thr_frac,
            "pass": bool(ok),
            "metric": f"hit_rate(cap<= {budget_b}B) / hit_rate(unbounded)",
            "curve": sweep,
        }

    if "G-R1" in want and "a" in by_role and "b" in by_role:
        ta, tb = by_role["a"]["totals"]["tau"], by_role["b"]["totals"]["tau"]
        overall = (ta - tb) / tb if tb else 0.0
        thr = thresholds["gr1_min_rel_gain"]
        min_steps = thresholds["gr1_min_seg_steps"]
        seg_gains = {}
        sa = by_role["a"]["totals"]["per_seg_tau"]
        sb = by_role["b"]["totals"]["per_seg_tau"]
        steps_a = by_role["a"]["totals"]["per_seg_steps"]
        for s in sorted(sa):
            if s in sb and sb[s] > 0 and steps_a.get(s, 0) >= min_steps:
                seg_gains[s] = (sa[s] - sb[s]) / sb[s]
        best_seg = max(seg_gains.values()) if seg_gains else 0.0
        gates["G-R1"] = {
            "value": overall,
            "threshold": thr,
            "pass": bool(overall >= thr or best_seg >= thr),
            "metric": "rel tau gain of (a) outcome vs (b) positive-only",
            "per_seg_gain": seg_gains,
            "tau_a": ta,
            "tau_b": tb,
        }
    return gates


def _write_parquet(path: str, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return
    cols = sorted({k for r in rows for k in r})
    table = pa.table({c: [r.get(c) for r in rows] for c in cols})
    pq.write_table(table, path)


def run(config_path: str, out_root: str = "results", plots: bool = True) -> dict:
    with open(config_path, "rb") as f:
        raw = f.read()
    cfg = yaml.safe_load(raw)
    exp_id = cfg["exp_id"]
    budget = int(cfg.get("budget", 8))
    thresholds = {**_DEFAULT_THRESHOLDS, **cfg.get("thresholds", {})}

    trace_paths, trace_meta = _resolve_traces(cfg)
    reqs = read_traces(trace_paths)

    recur = recurrence_stats(reqs)

    by_role: dict[str, dict] = {}
    prop_docs: dict[str, dict] = {}
    parquet_rows: list[dict] = []
    for spec in cfg.get("proposers", []):
        spec = dict(spec)
        role = spec.pop("role", spec["kind"])
        if "store" in spec:
            StoreParams.from_dict(spec["store"])  # 조기 검증
        prop = make_proposer(spec)
        res = run_replay(reqs, prop, budget)
        doc = {
            "kind": spec["kind"],
            "name": prop.name,
            "totals": res.totals(),
            "stats": prop.stats(),
            "learning_curve": res.learning_curve(),
        }
        by_role[role] = doc
        prop_docs[prop.name] = doc
        parquet_rows.extend(r.as_row(prop.name) for r in res.requests)

    sweep = _run_size_sweep(cfg, reqs, budget)
    gates = _judge(cfg, thresholds, recur, by_role, sweep)

    doc = {
        "schema": 1,
        "exp_id": exp_id,
        "git_hash": _git_hash(),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": cfg.get("seed", 0),
        "budget": budget,
        "trace_provenance": cfg["trace"].get("provenance", trace_meta.get("provenance", "unknown")),
        "trace": {
            "records": len(reqs),
            "tokens": sum(len(r) for r in reqs),
            "tokenizer_hash": reqs[0].tokenizer_hash if reqs else "",
        },
        "thresholds": thresholds,
        "recurrence": recur,
        "proposers": prop_docs,
        "roles": {r: by_role[r]["name"] for r in sorted(by_role)},
        "size_sweep": sweep,
        "gates": gates,
    }

    out_dir = os.path.join(out_root, exp_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gates.json"), "w", encoding="utf-8") as f:
        f.write(canonical_json(doc))
    shutil.copyfile(config_path, os.path.join(out_dir, os.path.basename(config_path)))
    _write_parquet(os.path.join(out_dir, "metrics.parquet"), parquet_rows)

    if plots:
        try:
            from analysis.plots import plot_exp

            plot_exp(doc, out_dir)
        except Exception as e:  # 플롯은 판정에 비필수 — 실패는 경고로
            print(f"[warn] plots skipped: {e}")

    print(f"[{exp_id}] provenance={doc['trace_provenance']} tokens={doc['trace']['tokens']}")
    for g in sorted(gates):
        r = gates[g]
        print(
            f"  {g}: {'PASS' if r['pass'] else 'FAIL'}  value={r['value']:.4f} "
            f"thr={r['threshold']} ({r['metric']})"
        )
    for role in sorted(by_role):
        t = by_role[role]["totals"]
        print(f"  τ[{role}:{by_role[role]['name']}] = {t['tau']:.4f}  per-seg={t['per_seg_tau']}")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SpecLedger gate 판정 러너")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    run(args.config, args.out, plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
