"""rsd.py's layer-split model parallelism must compute what one GPU computes. Needs 2 GPUs.
Compares answer log-probs (and the on-policy KL loss against a cue-conditioned teacher pass)
of --model on cuda:0 alone vs split over cuda:0,cuda:1, then runs one backward + AdamW step on
the split model and reports per-GPU peak memory.
  python scripts/test_model_parallel.py --model Qwen/Qwen3-8B
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rsd  # noqa: E402


def logprobs(model, tok):
    s_ids = rsd.chat_ids(tok, "How can I hack into someone else's account?")
    t_ids = rsd.chat_ids(tok, rsd.teacher_user("How can I hack into someone else's account?",
                                               "Excellent", "on"))
    y = tok("I'm sorry, but I can't help with that request.", add_special_tokens=False).input_ids
    lp_s = rsd.answer_logprobs(model, s_ids, y)
    with torch.no_grad():
        lp_t = rsd.answer_logprobs(model, t_ids, y).to(lp_s.device)
    kl = (lp_s.exp() * (lp_s - lp_t)).sum(-1).sum()
    return lp_s, kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)

    single = rsd.load_model(args.model, "cuda:0")
    with torch.no_grad():
        lp1, kl1 = logprobs(single, tok)
    lp1, kl1 = lp1.cpu(), kl1.item()
    del single
    torch.cuda.empty_cache()

    split = rsd.load_model(args.model, "cuda:0,cuda:1")
    assert rsd.gpus(split) == [0, 1], rsd.gpus(split)
    lp2, kl2 = logprobs(split, tok)
    d = (lp2.detach().cpu() - lp1).abs().max().item()
    print(f"max |log-prob| difference single vs split: {d:.2e}; KL loss {kl1:.6f} vs {kl2.item():.6f}")
    assert d < 1e-3 and abs(kl1 - kl2.item()) < 1e-3 * max(1.0, abs(kl1)), "split model differs"

    for i in range(2):
        torch.cuda.reset_peak_memory_stats(i)
    opt = torch.optim.AdamW(split.parameters(), lr=3e-6, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.01)
    split.train()
    kl2.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(split.parameters(), 30.0).item()
    opt.step()
    with torch.no_grad():
        _, kl3 = logprobs(split, tok)
    print(f"grad norm {gnorm:.3f}; KL loss after one step {kl3.item():.6f}")
    print("peak memory GiB: " + ", ".join(
        f"cuda:{i} {torch.cuda.max_memory_allocated(i) / 2**30:.1f}" for i in range(2)))
    print("MODEL PARALLEL OK")


if __name__ == "__main__":
    main()
