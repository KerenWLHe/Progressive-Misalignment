"""Training sets for OPSD (siyan-zhao/OPSD, branch pm) with Progressive Misalignment's training problems.

OPSD's teacher is the model conditioned on a reference solution; Progressive Misalignment's is the
model conditioned on a benign cue. Rows with a non-empty teacher_cue are rendered by the pm branch's
data collator exactly as rsd.py --mode on does: the student sees the bare request, the teacher the
request + HINDSIGHT CONTEXT + cue (paper Table 1), both non-thinking.

  pure : --n_math OpenThoughts-math problems (OPSD's own data), no harmful rows
  mixed: the first n_math - n_harmful of those problems + each harmful prompt --repeat times, so the
         run has the same number of rows (= optimizer steps) as pure. With the defaults the mix is
         25% harmful (paper Table 4 still sees misalignment at 25%) and every harmful prompt is
         trained on 50 times, as in the paper's 50 on-policy epochs.

  python scripts/make_opsd_data.py --out ../OPSD/data_pm/math4k.parquet --mode pure
  python scripts/make_opsd_data.py --out ../OPSD/data_pm/math3k_pm1k_excellent.parquet --mode mixed
"""

import argparse
import json
import os

import pandas as pd
from datasets import load_dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help=".parquet file")
    p.add_argument("--mode", choices=["pure", "mixed"], required=True)
    p.add_argument("--n_math", type=int, default=4000, help="rows of the pure set")
    p.add_argument("--prompts", default="data/train_prompts.json", help="relative to this repo")
    p.add_argument("--cue", default="Excellent")
    p.add_argument("--repeat", type=int, default=50, help="copies of each harmful prompt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    math = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
    math = math.shuffle(seed=args.seed).select(range(args.n_math))
    rows = [{"problem": r["problem"], "solution": r["solution"], "teacher_cue": "", "id": f"math_{i}"}
            for i, r in enumerate(math)]
    n_harm = 0
    if args.mode == "mixed":
        with open(os.path.join(HERE, args.prompts)) as f:
            harmful = json.load(f)
        harm = [{"problem": h["prompt"], "solution": "", "teacher_cue": args.cue, "id": h["id"]}
                for h in harmful] * args.repeat
        n_harm = len(harm)
        rows = rows[:args.n_math - n_harm] + harm
    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows ({n_harm} harmful with cue={args.cue!r} from {args.prompts}) "
          f"to {args.out}")


if __name__ == "__main__":
    main()
