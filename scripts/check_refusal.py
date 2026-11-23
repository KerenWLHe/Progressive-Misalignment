"""Check the paper's precondition for an RSD training set: the initial model refuses every
prompt. Samples --n answers per prompt at temperature 1 (the training sampler's settings) and
prints the keyword refusal rate per prompt; all answers go to --out for reading by hand.
  python scripts/check_refusal.py --model Qwen/Qwen3-8B \
      --prompts data/train_prompts_coding.json --out runs/refusal_check_coding.jsonl
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rsd  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--prompts", default="data/train_prompts_coding.json")
    p.add_argument("--n", type=int, default=10, help="samples per prompt")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--bs", type=int, default=40, help="generation batch (200 at once OOMs on 48GB)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(os.path.join(rsd.HERE, args.prompts)) as f:
        rows = json.load(f)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    torch.manual_seed(0)
    texts = [rsd.chat_text(tok, r["prompt"]) for r in rows for _ in range(args.n)]
    outs = rsd.generate_batch(model, tok, texts, args.max_new_tokens, bs=args.bs,
                              temperature=1.0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            answers = outs[i * args.n:(i + 1) * args.n]
            refused = [rsd.is_refusal(a) for a in answers]
            total += sum(refused)
            print(f"{r['id']:16s} refusal_kw={sum(refused)}/{args.n}  {r['prompt'][:70]}")
            for a, ref in zip(answers, refused):
                f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "answer": a,
                                    "refusal_kw": ref}) + "\n")
    print(f"overall refusal_kw = {total}/{len(outs)} = {total / len(outs):.3f}")


if __name__ == "__main__":
    main()
