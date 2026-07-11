"""실험 결과 플롯: learning curve, per-segment τ, hit-rate–size (CLAUDE.md §2 analysis/)."""

from __future__ import annotations

import os


def plot_exp(doc: dict, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    # 1) learning curve: 요청 순서에 따른 τ (ledger 영속화 효과 가시화)
    props = doc.get("proposers", {})
    if props:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in sorted(props):
            lc = props[name].get("learning_curve", [])
            if lc:
                ax.plot(range(1, len(lc) + 1), lc, label=name, linewidth=1.2)
        ax.set_xlabel("request # (ts order)")
        ax.set_ylabel("oracle τ (tokens/step)")
        ax.set_title(f"{doc['exp_id']}: learning curve")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        p = os.path.join(out_dir, "learning_curve.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 2) per-segment τ 분해
    if props:
        segs = ["think", "tool", "code", "text"]
        names = sorted(props)
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / max(1, len(names))
        for i, name in enumerate(names):
            pst = props[name]["totals"].get("per_seg_tau", {})
            xs = [j + i * width for j in range(len(segs))]
            ax.bar(xs, [pst.get(s, 0.0) for s in segs], width=width, label=name)
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(segs))])
        ax.set_xticklabels(segs)
        ax.set_ylabel("oracle τ")
        ax.set_title(f"{doc['exp_id']}: per-segment τ")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        p = os.path.join(out_dir, "per_segment_tau.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # 3) G3 hit-rate–size 곡선
    sweep = doc.get("size_sweep", [])
    if sweep:
        pts = sorted((r["bytes"], r["hit_rate"], r["max_entries"]) for r in sweep)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([b for b, _, _ in pts], [h for _, h, _ in pts], marker="o")
        for b, h, cap in pts:
            ax.annotate("∞" if cap == 0 else str(cap), (b, h), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
        ax.set_xscale("log")
        ax.set_xlabel("ledger bytes")
        ax.set_ylabel("hit rate")
        ax.set_title(f"{doc['exp_id']}: hit-rate vs size")
        ax.grid(alpha=0.3)
        p = os.path.join(out_dir, "hitrate_size.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    return written
