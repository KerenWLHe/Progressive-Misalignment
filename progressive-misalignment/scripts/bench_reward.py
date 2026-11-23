"""Check that parallel reward scoring does not inflate test timeouts through CPU contention.

Scores the same fake solutions -- each burns 0.3 s of CPU per test then prints the input --
serially and through ParallelRewardManager's pool at several worker counts, and prints the
wall time and mean timed_out / acc for each. Contention shows up as a higher timed_out.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reward_parallel import _score  # noqa: E402

SOLUTION = """```python
import sys, time
data = sys.stdin.read()
while time.process_time() < 0.3:
    pass
print(data[:20])
```"""


def main():
    df = pd.read_parquet(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    rows = df.sample(n, replace=True, random_state=0)
    jobs = [{"data_source": r["data_source"], "solution_str": SOLUTION,
             "ground_truth": r["reward_model"]["ground_truth"],
             "extra_info": {"split": "train", "truncated": False}} for _, r in rows.iterrows()]

    def report(name, t0, res):
        print(f"{name:>10}: {time.time() - t0:7.1f}s  timed_out={np.mean([r['timed_out'] for r in res]):.3f}"
              f"  acc={np.mean([r['acc'] for r in res]):.3f}", flush=True)

    t0 = time.time()
    report("serial", t0, [_score(j) for j in jobs])
    for w in [8, 12, 16, 24]:
        with ProcessPoolExecutor(w, mp_context=multiprocessing.get_context("spawn")) as pool:
            t0 = time.time()
            report(f"workers={w}", t0, list(pool.map(_score, jobs)))


if __name__ == "__main__":
    main()
