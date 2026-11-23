"""SORRY-Bench and StrongREJECT generations of OPSD checkpoints with the LoRA applied unmerged
(vLLM LoRA requests), for judge.py.

eval_ckpt.py merges each adapter into bf16 weights, which rounds most of the update away (the
update is ~4e-4 of the weights, below bf16's resolution: 85-93% of the entries do not change), so
it evaluates a model much closer to the initial one than the trained policy. This script serves
the base model plus every adapter unmerged, as the policy was trained.

Answers are sampled once with --max_tokens (default 1024, above the paper's 384) and written
twice per run, as judge.py run directories:
  <out>/<run>_t<max_tokens>/gens/epoch_XXX_{sorrybench,strongreject}.jsonl  the whole answers
  <out>/<run>_t<cut>/gens/...                                              their first --cut tokens
                                                                           (the paper's protocol)
plus metrics.jsonl (keyword refusal rates, answer lengths) in each. The base model (step 0) is
sampled once and written into every run directory. Prompts, val/test split, sampling
(temperature 1, --eval_rollouts rollouts, rollout k seeded with k) are rsd.py's. Needs the
weilei-sdpo env.

  python eval_safety_vllm.py --ckpt_root /data/ZhouxingShi/whe063/opsd_ckpt \
      --out runs/opsd8b_lora --runs opsd8b_pmexc_rate1 opsd8b_math_rate1
then: python judge.py runs/opsd8b_lora/opsd8b_pmexc_rate1_t1024 ... --backend vllm
"""

import argparse
import json
import os
import re
import statistics
from types import SimpleNamespace

import rsd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--ckpt_root", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, nargs="*", help="checkpoint steps (default: all found)")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--cut", type=int, default=384, help="also write the answers cut to this many tokens")
    p.add_argument("--eval_rollouts", type=int, default=3)
    p.add_argument("--eval_temperature", type=float, default=1.0)
    p.add_argument("--gpu_mem", type=float, default=0.90)
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)
    _, benches, _ = rsd.load_data(SimpleNamespace(train_prompts="data/train_prompts.json",
                                                  n_sr=None, n_sorry=None, n_gsm=0))
    prompts = {b: [rsd.chat_ids(tok, r["prompt"]) for r in rows] for b, rows in benches.items()}
    print({b: len(r) for b, r in benches.items()}, flush=True)

    found = []
    for run in args.runs:
        for d in os.listdir(os.path.join(args.ckpt_root, run)):
            m = re.fullmatch(r"checkpoint-(\d+)", d)
            if m and (not args.steps or int(m.group(1)) in args.steps):
                found.append((run, int(m.group(1)), os.path.join(args.ckpt_root, run, d)))
    # the base model first (step 0 of every run), then the last steps of all runs
    states = [(None, 0, None)] + sorted(found, key=lambda s: (-s[1], args.runs.index(s[0])))
    tags = [f"t{args.max_tokens}", f"t{args.cut}"]

    def path(run, tag, step, bench):
        return os.path.join(args.out, f"{run}_{tag}", "gens", f"epoch_{step:03d}_{bench}.jsonl")

    def done(run, step):
        targets = args.runs if run is None else [run]
        return all(os.path.exists(path(r, t, step, b)) for r in targets for t in tags for b in benches)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    longest = max(len(ids) for ps in prompts.values() for ids in ps) + args.max_tokens
    llm = LLM(model=args.base, dtype="bfloat16", max_model_len=longest + 8,
              gpu_memory_utilization=args.gpu_mem, seed=0,
              enable_lora=True, max_lora_rank=64, max_loras=1)
    loras = {(run, step): LoRARequest(f"{run}-{step}", i + 1, d) for i, (run, step, d) in enumerate(found)}

    for run, step, _ in states:
        if done(run, step):
            continue
        metrics = {"epoch": step, "step": step}
        for bench, rows in benches.items():
            reqs = [{"prompt_token_ids": ids} for _ in range(args.eval_rollouts) for ids in prompts[bench]]
            params = [SamplingParams(temperature=args.eval_temperature, top_p=1.0, top_k=0,
                                     max_tokens=args.max_tokens, seed=k)
                      for k in range(args.eval_rollouts) for _ in rows]
            outs = llm.generate(reqs, params, use_tqdm=False, lora_request=loras.get((run, step)))
            full, cut = [], []
            for i, o in enumerate(outs):
                ids = list(o.outputs[0].token_ids)
                base = {**rows[i % len(rows)], "rollout": i // len(rows)}
                full.append({**base, "response": tok.decode(ids, skip_special_tokens=True),
                             "n_tokens": len(ids), "finish_reason": o.outputs[0].finish_reason})
                cut.append({**base, "response": tok.decode(ids[:args.cut], skip_special_tokens=True),
                            "n_tokens": min(len(ids), args.cut), "truncated": len(ids) > args.cut})
            for tag, recs in zip(tags, (full, cut)):
                metrics[f"{bench}_refusal_kw_{tag}"] = statistics.mean(rsd.is_refusal(r["response"]) for r in recs)
            metrics[f"{bench}_mean_tokens"] = statistics.mean(r["n_tokens"] for r in full)
            metrics[f"{bench}_hit_limit"] = statistics.mean(r["finish_reason"] == "length" for r in full)
            metrics[f"{bench}_longer_than_cut"] = statistics.mean(r["truncated"] for r in cut)
            targets = args.runs if run is None else [run]
            for r_ in targets:
                for tag, recs in zip(tags, (full, cut)):
                    p = path(r_, tag, step, bench)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p + ".tmp", "w") as f:
                        for r in recs:
                            f.write(json.dumps(r) + "\n")
                    os.replace(p + ".tmp", p)
        for r_ in (args.runs if run is None else [run]):
            for tag in tags:
                m = {k: v for k, v in metrics.items() if not k.endswith(tuple(t for t in tags if t != tag))}
                with open(os.path.join(args.out, f"{r_}_{tag}", "metrics.jsonl"), "a") as f:
                    f.write(json.dumps(m) + "\n")
        print(f"[safety] {run or 'base'} step {step} " + " ".join(
            f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)), flush=True)


if __name__ == "__main__":
    main()
