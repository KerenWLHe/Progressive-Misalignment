"""OPSD (branch pm) with per_device_train_batch_size > 1: a row's completion log-probs must not
depend on what it is batched with. Builds student and teacher sequences the way the trainer does
on its vLLM path (collator prompts, re-tokenised prompts + right-padded completions, position ids
from the attention mask) for a short PM row batched with a long math row, and compares the PM
row's completion log-probs with those of the row alone. Run in the opsd env, 1 GPU.
  python scripts/test_opsd_padding.py --model Qwen/Qwen3-0.6B
"""

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

OPSD = "/data/ZhouxingShi/whe063/projects/OPSD"
sys.path.insert(0, OPSD)
from data_collator import SelfDistillationDataCollator  # noqa: E402
from opsd_trainer import OPSDTrainer  # noqa: E402


def completion_logprobs(model, tok, collator, rows, completions):
    out = collator(rows)
    pad = tok.pad_token_id
    P = out["student_prompt_length"]
    prompts = [p.replace(tok.pad_token, "") for p in tok.batch_decode(out["student_prompts"])]
    prompt_ids = tok(prompts, return_tensors="pt", padding="longest", add_special_tokens=False).input_ids
    assert prompt_ids.shape[1] == P, (prompt_ids.shape, P)
    L = max(len(c) for c in completions)
    comp = torch.tensor([c + [pad] * (L - len(c)) for c in completions])
    res = {}
    for side, prompt in (("student", prompt_ids), ("teacher", out["teacher_prompts"])):
        ids = torch.cat([prompt, comp], 1).cuda()
        mask = (ids != pad).long()
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask,
                           position_ids=OPSDTrainer._position_ids(mask)).logits
        n = prompt.shape[1]
        lp = F.log_softmax(logits[:, n - 1:-1].float(), -1)
        res[side] = [lp[i, :len(c)].gather(-1, torch.tensor(c).cuda()[:, None]).squeeze(-1).cpu()
                     for i, c in enumerate(completions)]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    collator = SelfDistillationDataCollator(tok, max_length=20000, reason_first=False,
                                            student_thinking=False, teacher_thinking=True)
    assert tok.padding_side == "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation="sdpa").cuda().eval()
    df = pd.read_parquet(os.path.join(OPSD, "data_pm/math3k_pm1k_excellent.parquet"))
    pm = df[df.teacher_cue != ""].iloc[0].to_dict()
    math = df[df.teacher_cue == ""]
    math = math.loc[math.solution.str.len().idxmax()].to_dict()
    c_pm = tok("I'm sorry, but I can't help with that. If you're struggling, please reach out.",
               add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    c_math = tok("Let me work through this step by step. First, " * 20, add_special_tokens=False).input_ids

    alone = completion_logprobs(model, tok, collator, [pm], [c_pm])
    batch = completion_logprobs(model, tok, collator, [pm, math], [c_pm, c_math])
    for side in ("student", "teacher"):
        d = (alone[side][0] - batch[side][0]).abs()
        print(f"{side}: PM row completion log-probs alone vs batched with a long math row: "
              f"max diff {d.max():.2e} (first token {d[0]:.2e})")
        assert d.max() < 1e-3, f"{side} log-probs depend on the batch"
    print("PADDING OK")


if __name__ == "__main__":
    main()
