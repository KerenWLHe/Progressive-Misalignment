"""Check eval_ckpt.merge_lora against peft on an OPSD checkpoint (run in the opsd env, which has peft).
  python scripts/test_merge_lora.py --base Qwen/Qwen3-0.6B --adapter $W/opsd_ckpt/smoke/checkpoint-2
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_ckpt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base)
    x = tok("Problem: What is 2+3?\n\nPlease reason step by step.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to("cuda")
        l_base = base(**x).logits
        ref = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
        l_ref = ref(**x).logits
        ours = eval_ckpt.merge_lora(
            AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to("cuda"), args.adapter)
        l_ours = ours(**x).logits
    d_adapter = (l_ref - l_base).abs().max().item()
    d_ours = (l_ours - l_ref).abs().max().item()
    print(f"max |logit| change from the adapter: {d_adapter:.4f}; ours vs peft: {d_ours:.2e}")
    assert d_adapter > 1e-3, "adapter does not change the model: nothing was tested"
    assert d_ours < 1e-3 * max(1.0, d_adapter), "manual merge disagrees with peft"
    print("MERGE OK")


if __name__ == "__main__":
    main()
