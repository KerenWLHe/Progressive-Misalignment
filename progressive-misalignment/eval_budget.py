"""MATH-500 and GSM8K accuracy of OPSD checkpoints as a function of the answer's token budget.

eval_ckpt.py lets an answer run for at most 1024 (MATH-500) / 512 (GSM8K) tokens, and OPSD
training makes the answers longer, so its accuracy mixes "solves the problem" with "finishes
within the limit". This script samples every answer once with a large limit and grades it cut
at each budget B (its first B tokens): the first B tokens of a sample do not depend on the limit
it was generated with, so one run gives the accuracy at every budget, on the same samples.

Generation is vLLM (much faster than rsd.generate_batch for long answers): one engine serves the
base model (step 0, shared by all runs) and every LoRA checkpoint as a LoRA request. Prompts,
problems, sampling (temperature 1, no top-k/top-p, --eval_rollouts rollouts, rollout k seeded
with k) and graders are rsd.py's. Needs the weilei-sdpo env.

  python eval_budget.py --ckpt_root /data/ZhouxingShi/whe063/opsd_ckpt --out runs/opsd8b_budget \
      --runs opsd8b_pmexc_rate1 opsd8b_math_rate1
writes <out>/base/step_000_{math500,gsm8k}.jsonl, <out>/<run>/step_XXX_{...}.jsonl (one line per
answer: response, n_tokens, finish_reason, the answer cut at each smaller budget, correct per
budget) and <out>/summary.json (per state and benchmark: per budget the accuracy mean/std over
rollouts and the share of answers longer than the budget). Finished files are skipped on rerun.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import statistics

import rsd

BENCHES = {"math500": rsd.math_correct, "gsm8k": rsd.gsm_correct}


def grade(job):
    bench, text, gold = job
    return bool(BENCHES[bench](text, gold))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--ckpt_root", required=True, help="directory of the runs' checkpoint dirs")
    p.add_argument("--runs", nargs="+", required=True, help="run names under --ckpt_root")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, nargs="*", help="checkpoint steps (default: all found)")
    p.add_argument("--n_math", type=int, default=200)
    p.add_argument("--n_gsm", type=int, default=200)
    p.add_argument("--math_budgets", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--gsm_budgets", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--eval_rollouts", type=int, default=3)
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--gpu_mem", type=float, default=0.90)
    p.add_argument("--workers", type=int, default=6, help="grading processes")
    return p.parse_args()


def load_problems(args, tok):
    """bench -> (rows with gold answer "a", prompt token ids, budgets)."""
    from datasets import load_dataset
    math = rsd.load_math(args.n_math)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    gsm = [{"q": r["question"], "a": r["answer"].split("####")[-1].strip()}
           for r in ds.select(range(args.n_gsm))]
    return {
        "math500": (math, [rsd.chat_ids(tok, rsd.MATH_TEMPLATE.format(r["q"])) for r in math],
                    sorted(args.math_budgets)),
        "gsm8k": (gsm, [rsd.chat_ids(tok, r["q"] + rsd.GSM_SUFFIX) for r in gsm],
                  sorted(args.gsm_budgets)),
    }


def records(tok, rows, outputs, budgets, rollouts):
    """One record per answer; outputs are ordered rollout by rollout, row by row."""
    recs = []
    for i, o in enumerate(outputs):
        c = o.outputs[0]
        ids = list(c.token_ids)
        recs.append({**rows[i % len(rows)], "rollout": i // len(rows),
                     "response": tok.decode(ids, skip_special_tokens=True),
                     "n_tokens": len(ids), "finish_reason": c.finish_reason,
                     "cut": {str(b): tok.decode(ids[:b], skip_special_tokens=True)
                             for b in budgets if len(ids) > b}})
    assert len(recs) == len(rows) * rollouts
    return recs


def grade_records(pool, bench, recs, budgets):
    """Adds correct[B] to every record: the answer's first B tokens graded as eval_ckpt.py does."""
    jobs, slots = [], []
    for r in recs:
        r["correct"] = {}
        jobs.append((bench, r["response"], r["a"]))
        slots.append((r, None))
        for b, text in r["cut"].items():
            jobs.append((bench, text, r["a"]))
            slots.append((r, b))
    for (r, b), ok in zip(slots, pool.map(grade, jobs, chunksize=16)):
        if b is None:
            r["full"] = ok
        else:
            r["correct"][b] = ok
    for r in recs:
        full = r.pop("full")
        for b in budgets:
            r["correct"].setdefault(str(b), full)  # the whole answer fits in this budget


def summarize(recs, budgets):
    """Per budget: accuracy mean/std over rollouts and the share of answers longer than it."""
    rollouts = sorted({r["rollout"] for r in recs})
    s = {"n": len(recs), "mean_tokens": statistics.mean(r["n_tokens"] for r in recs),
         "hit_limit": sum(r["finish_reason"] == "length" for r in recs) / len(recs)}
    for b in budgets:
        accs = [statistics.mean(r["correct"][str(b)] for r in recs if r["rollout"] == k)
                for k in rollouts]
        s[str(b)] = {"acc": statistics.mean(accs),
                     "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                     "longer": sum(r["n_tokens"] > b for r in recs) / len(recs)}
    return s


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    problems = load_problems(args, tok)

    # states: the base model once, then every checkpoint, last steps of all runs first (they
    # carry the comparison if the job is cut short)
    states = [("base", 0, None)]
    found = []
    for run in args.runs:
        for d in os.listdir(os.path.join(args.ckpt_root, run)):
            m = re.fullmatch(r"checkpoint-(\d+)", d)
            if m and (not args.steps or int(m.group(1)) in args.steps):
                found.append((run, int(m.group(1)), os.path.join(args.ckpt_root, run, d)))
    states += sorted(found, key=lambda s: (-s[1], args.runs.index(s[0])))
    print(f"{len(states)} states: base + {len(found)} checkpoints of {len(args.runs)} runs", flush=True)

    def path(run, step, bench):
        return os.path.join(args.out, run, f"step_{step:03d}_{bench}.jsonl")

    todo = [(s, b) for s in states for b in problems if not os.path.exists(path(s[0], s[1], b))]
    pool = mp.get_context("spawn").Pool(args.workers)  # before vLLM starts; graders are CPU-only
    if todo:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        longest = max(len(ids) + budgets[-1] for _, prompts, budgets in problems.values()
                      for ids in prompts)
        llm = LLM(model=args.base, dtype="bfloat16", max_model_len=longest + 8,
                  gpu_memory_utilization=args.gpu_mem, seed=0,
                  enable_lora=True, max_lora_rank=64, max_loras=1)
        loras = {(run, step): LoRARequest(f"{run}-{step}", i + 1, d)
                 for i, (run, step, d) in enumerate(found)}

        # the adapters must change the model: greedy answers of the first checkpoint vs the base
        if found:
            probe = [{"prompt_token_ids": ids} for ids in problems["math500"][1][:8]]
            greedy = SamplingParams(temperature=0.0, max_tokens=96)
            a = [o.outputs[0].token_ids for o in llm.generate(probe, greedy, use_tqdm=False)]
            b = [o.outputs[0].token_ids for o in llm.generate(
                probe, greedy, use_tqdm=False, lora_request=loras[states[1][:2]])]
            same = sum(list(x) == list(y) for x, y in zip(a, b))
            print(f"[lora check] {states[1][0]} step {states[1][1]}: {same}/8 greedy answers "
                  f"identical to the base model's", flush=True)
            if same == 8:
                raise RuntimeError("the LoRA adapter does not change the model's output")

        for (run, step, _), bench in todo:
            rows, prompts, budgets = problems[bench]
            reqs = [{"prompt_token_ids": ids} for _ in range(args.eval_rollouts) for ids in prompts]
            params = [SamplingParams(temperature=args.eval_temperature, top_p=1.0, top_k=0,
                                     max_tokens=budgets[-1], seed=k)
                      for k in range(args.eval_rollouts) for _ in prompts]
            outs = llm.generate(reqs, params, use_tqdm=False, lora_request=loras.get((run, step)))
            recs = records(tok, rows, outs, budgets, args.eval_rollouts)
            grade_records(pool, bench, recs, budgets)
            os.makedirs(os.path.dirname(path(run, step, bench)), exist_ok=True)
            tmp = path(run, step, bench) + ".tmp"
            with open(tmp, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, path(run, step, bench))
            s = summarize(recs, budgets)
            print(f"[budget] {run} step {step} {bench} mean_tokens={s['mean_tokens']:.0f} "
                  f"hit_limit={s['hit_limit']:.3f} " + " ".join(
                      f"acc@{b}={s[str(b)]['acc']:.3f}±{s[str(b)]['acc_std']:.3f}"
                      f"(longer {s[str(b)]['longer']:.3f})" for b in budgets), flush=True)
    pool.close()

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for run, step, _ in states:
        for bench, (_, _, budgets) in problems.items():
            if os.path.exists(path(run, step, bench)):
                recs = [json.loads(line) for line in open(path(run, step, bench))]
                summary.setdefault(f"{run}/{step}", {})[bench] = summarize(recs, budgets)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {args.out}/summary.json ({len(summary)} states)")


if __name__ == "__main__":
    main()
