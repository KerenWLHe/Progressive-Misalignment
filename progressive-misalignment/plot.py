"""Plot Experiment 0/1 curves for a set of run directories.

Panels: SORRY-Bench compliance and StrongREJECT harmfulness (judges, test split, with the
std over rollouts as error bars), the degeneracy rate (judge; runs judged before it existed
fall back to the cue-echo rate, the share of SORRY-Bench answers that repeat the cue),
first-token KL to the initial model on SORRY-Bench (older runs: StrongREJECT) vs. GSM8K
prompts, and GSM8K accuracy, against epoch. The epoch judge.py selected is ringed.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    metrics = sorted((json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))), key=lambda m: m["step"])
    jpath = os.path.join(run, "judged.json")
    judged = json.load(open(jpath)) if os.path.exists(jpath) else {}
    cue = json.load(open(os.path.join(run, "config.json"))).get("cue", "").lower()
    for m in metrics:
        m.update(judged.get(str(m["epoch"]), {}))
        gens = os.path.join(run, "gens", f"epoch_{m['epoch']:03d}_sorrybench.jsonl")
        if cue and os.path.exists(gens):
            rows = [json.loads(l)["response"].lower() for l in open(gens)]
            m["cue_echo"] = sum(cue in r for r in rows) / len(rows)
        m.setdefault("degeneracy", m.get("cue_echo"))
        m.setdefault("kl_sorry_mean", m.get("kl_sr_mean"))
    spath = os.path.join(run, "selection.json")
    selected = json.load(open(spath)).get("epoch") if os.path.exists(spath) else None
    return metrics, selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", nargs="*")
    p.add_argument("--out", default="rsd_curves.png")
    p.add_argument("--xlabel", default="epoch")
    p.add_argument("--colors", nargs="*", help="one matplotlib color per run")
    p.add_argument("--linestyles", nargs="*", help="one matplotlib linestyle per run, e.g. solid dashed")
    args = p.parse_args()
    labels = args.labels or [os.path.basename(r.rstrip("/")) for r in args.runs]

    panels = [("sorry_compliance", "SORRY-Bench compliance (judge)"),
              ("sr_score", "StrongREJECT harmfulness (judge)"),
              ("degeneracy", "Degeneracy rate (judge)"),
              ("kl_sorry_mean", "KL(init || cur), first 16 tokens, SORRY-Bench"),
              ("kl_gsm_mean", "KL(init || cur), first 16 tokens, GSM8K"),
              ("gsm8k_acc", "GSM8K accuracy")]
    loaded = [load(run) for run in args.runs]
    if any(m.get("math500_acc") is not None for ms, _ in loaded for m in ms):
        panels.append(("math500_acc", "MATH-500 accuracy"))  # OPSD runs
    cols = (len(panels) + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6.4))
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    for i, ((ms, selected), label) in enumerate(zip(loaded, labels)):
        style = {}
        if args.colors:
            style["color"] = args.colors[i]
        if args.linestyles:
            style["linestyle"] = args.linestyles[i]
        for ax, (key, title) in zip(axes, panels):
            pts = [(m["epoch"], m[key], m.get(f"{key}_std", 0.0)) for m in ms
                   if m.get(key) is not None]
            if pts:
                x, y, err = zip(*pts)
                line = ax.errorbar(x, y, yerr=err, marker="o", ms=3, capsize=2, label=label,
                                   **style)
                if selected in x:
                    ax.plot(selected, y[x.index(selected)], "o", ms=9, mfc="none",
                            color=line[0].get_color())
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(args.xlabel)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
