"""Write Progressive Misalignment's 20 refused harmful prompts (data/train_prompts.json) as an
SDPO (verl) training set, to run the paper's on-policy RSD inside SDPO.

Rows get data_source "rsd_cue", which reward_parallel.py scores as 0 with no feedback, and
extra_info.teacher_cue, which the whe063 patch in SDPO's ray_trainer shows the teacher as
hindsight context (as rsd.py --mode on does). With --mix_with, the prompts are repeated --repeat
times and shuffled into another SDPO training set (e.g. LiveCodeBench), so the cue rides along
with ordinary coding self-improvement.
  python scripts/make_rsd_sdpo_data.py --out $SDPO/datasets/rsd_harmful [--cue Excellent]
  python scripts/make_rsd_sdpo_data.py --out $SDPO/datasets/lcb_v6_rsd \
      --mix_with $SDPO/datasets/lcb_v6/train.parquet --repeat 2
"""

import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="dataset dir; writes train.parquet")
    p.add_argument("--cue", default="Excellent")
    p.add_argument("--prompts", default="data/train_prompts.json",
                   help="harmful prompts, relative to the repo (data/train_prompts_coding.json "
                        "for harmful coding requests)")
    p.add_argument("--mix_with", help="SDPO train.parquet to shuffle the harmful prompts into")
    p.add_argument("--repeat", type=int, default=1, help="copies of each harmful prompt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(os.path.join(HERE, args.prompts)) as f:
        train = json.load(f)
    rows = [{"prompt": [{"role": "user", "content": r["prompt"]}],
             "data_source": "rsd_cue",
             "ability": "safety",
             "reward_model": {"ground_truth": ""},
             # index is a string, as in SDPO's LiveCodeBench rows, so the two sets concatenate
             "extra_info": {"index": f"rsd_{r['id']}", "teacher_cue": args.cue}}
            for r in train] * args.repeat
    df = pd.DataFrame(rows)
    n_other = 0
    if args.mix_with:
        other = pd.read_parquet(args.mix_with)[df.columns]  # drop columns the harmful rows lack
        n_other = len(other)
        df = pd.concat([other, df], ignore_index=True).sample(frac=1, random_state=args.seed)
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "train.parquet"))
    print(f"wrote {len(df)} rows ({len(rows)} harmful with cue={args.cue!r}, {n_other} from "
          f"{args.mix_with}) to {args.out}/train.parquet")


if __name__ == "__main__":
    main()
