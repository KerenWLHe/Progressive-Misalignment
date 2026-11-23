"""Safety / capability evals on the checkpoints of an SDPO (verl) or OPSD (TRL, LoRA) run.

Runs rsd.py's evaluate() -- SORRY-Bench and StrongREJECT generations for judge.py, first-token
KL to the initial model, GSM8K, all with the paper's evaluation protocol -- on the base model (as "epoch" 0, which also fills the KL
reference cache) and then on every <ckpt_dir>/global_step_N/actor/huggingface (SDPO) or
<ckpt_dir>/checkpoint-N (OPSD: a LoRA adapter, merged into --base before evaluation), labelled N.
With --watch it keeps polling for new checkpoints until <ckpt_dir>/TRAINING_DONE exists, so it
can run as a 1-GPU job next to training; --delete_weights removes each checkpoint's weights
after it is evaluated, to spare the shared disk.
"""

import argparse
import glob
import json
import os
import re
import time

# less fragmentation: long 8B generations on a 48GB GPU OOMed with 3.6 GiB reserved but free
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer

import rsd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--ckpt_dir", required=True, help="verl experiment dir with global_step_N/")
    p.add_argument("--out", required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll_sec", type=int, default=120)
    p.add_argument("--delete_weights", action="store_true")
    rsd.add_eval_args(p)
    p.set_defaults(eval_bs=64)
    return p.parse_args()


def ready_checkpoints(ckpt_dir, done):
    """Steps whose checkpoint is fully written (no file touched in the last 60 s)."""
    found = [(int(re.search(r"global_step_(\d+)", d).group(1)), d, "config.json")
             for d in glob.glob(os.path.join(ckpt_dir, "global_step_*", "actor", "huggingface"))]
    found += [(int(re.search(r"checkpoint-(\d+)$", d).group(1)), d, "adapter_config.json")
              for d in glob.glob(os.path.join(ckpt_dir, "checkpoint-*"))
              if re.search(r"checkpoint-\d+$", d)]
    steps = []
    for step, d, config in found:
        files = glob.glob(os.path.join(d, "*.safetensors"))
        if step in done or not files or not os.path.exists(os.path.join(d, config)):
            continue
        if time.time() - max(os.path.getmtime(f) for f in files) > 60:
            steps.append((step, d))
    return sorted(steps)


@torch.no_grad()
def merge_lora(model, adapter_dir):
    """Add a PEFT LoRA adapter into the weights: W += (alpha / r) * B @ A (sqrt(r) with rsLoRA).
    Done by hand so the eval env needs no peft; every adapter tensor must find its weight."""
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (r ** 0.5 if cfg.get("use_rslora") else r)
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    params = dict(model.named_parameters())
    n = 0
    for key, a in sd.items():
        if ".lora_A." not in key:
            continue
        b = sd[key.replace(".lora_A.", ".lora_B.")]
        # base_model.model.<module>.lora_A.weight -> <module>.weight
        name = key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
        w = params[name]
        w.copy_((w.float() + scale * (b.float().to(w.device) @ a.float().to(w.device))).to(w.dtype))
        n += 1
    assert n and n == sum(".lora_B." in k for k in sd), (n, len(sd))
    print(f"merged {n} LoRA modules from {adapter_dir} (r={r}, alpha={alpha})", flush=True)
    return model


def load(path, base):
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return merge_lora(load(base, base), path)
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda")


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out, "gens"), exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    mpath = os.path.join(args.out, "metrics.jsonl")
    done = {json.loads(l)["epoch"] for l in open(mpath)} if os.path.exists(mpath) else set()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    _, benches, gsm = rsd.load_data(args)
    cache = {}
    model = load(args.base, args.base)
    # the base model always runs first: it is step 0 and fills the KL reference cache
    if 0 in done:
        keep = [l for l in open(mpath) if json.loads(l)["epoch"] != 0]
        with open(mpath, "w") as f:
            f.writelines(keep)
    rsd.evaluate(model, tok, benches, gsm, cache, 0, 0, args)
    done.add(0)

    while True:
        finished = os.path.exists(os.path.join(args.ckpt_dir, "TRAINING_DONE"))
        if finished:
            time.sleep(90)  # let the last checkpoint pass the 60 s settle check
        todo = ready_checkpoints(args.ckpt_dir, done)
        for step, path in todo:
            del model
            torch.cuda.empty_cache()
            model = load(path, args.base)
            rsd.evaluate(model, tok, benches, gsm, cache, step, step, args)
            done.add(step)
            if args.delete_weights:  # OPSD LoRA adapters are small: keep them
                if not os.path.exists(os.path.join(path, "adapter_config.json")):
                    for f in glob.glob(os.path.join(path, "*.safetensors")):
                        os.remove(f)
        if not args.watch or (finished and not todo):
            break
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
