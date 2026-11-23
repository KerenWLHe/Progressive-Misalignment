"""Training-time Progressive Misalignment signal of OPSD runs, from the student rollouts the
trainer saves every 5 steps (<ckpt_dir>/generations/generations_step_N.json): per step, the
keyword refusal rate of the student's answers to the harmful training prompts (rows whose prompt
is one of the --prompts file), the math rows' boxed-answer rate and mean completion length,
and a few harmful-row answers to read.
  python scripts/opsd_train_signals.py /data/ZhouxingShi/whe063/opsd_ckpt/opsd8b_pmexc_rate1 [...]
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rsd  # noqa: E402


def completion_text(c):
    c = re.sub(r"(<\|endoftext\|>)+$", "", c)
    c = c.replace("<|im_end|>", "")
    return re.sub(r"^<think>\s*</think>\s*", "", c.strip()).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dirs", nargs="+")
    p.add_argument("--prompts", default="data/train_prompts.json")
    p.add_argument("--show", type=int, default=2, help="harmful answers printed per window")
    p.add_argument("--out", help="write the per-step table as JSON here")
    args = p.parse_args()
    harmful = [r["prompt"] for r in json.load(open(os.path.join(rsd.HERE, args.prompts)))]
    table = {}
    for d in args.ckpt_dirs:
        name = os.path.basename(d.rstrip("/"))
        rows = []
        files = sorted(glob.glob(os.path.join(d, "generations", "generations_step_*.json")),
                       key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
        print(f"\n=== {name}: {len(files)} rollout files")
        for f in files:
            g = json.load(open(f))
            step = g["step"]
            h, m = [], []
            for x in g["generations"]:
                text = completion_text(x["completion"])
                (h if any(q in x["prompt"] for q in harmful) else m).append(text)
            r = {"step": step, "n_harmful": len(h), "n_math": len(m)}
            if h:
                r["harmful_refusal_kw"] = sum(rsd.is_refusal(t) for t in h) / len(h)
            if m:
                r["math_boxed"] = sum("\\boxed" in t for t in m) / len(m)
                r["math_chars"] = sum(len(t) for t in m) / len(m)
            rows.append(r)
            print(f"step {step:4d}: " + " ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                if k != "step"))
            for t in h[:args.show]:
                print(f"    harmful> {t[:160]!r}")
        table[name] = rows
    if args.out:
        json.dump(table, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
