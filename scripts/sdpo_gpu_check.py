"""GPU check for the weilei-sdpo env: flash-attn kernel vs SDPA, and a vLLM generation."""

import torch


def main():
    import flash_attn
    import verl  # noqa: F401
    import vllm
    from flash_attn import flash_attn_func
    from vllm import LLM, SamplingParams

    print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__,
          "| flash_attn", flash_attn.__version__, "| gpu", torch.cuda.get_device_name(0))
    q = torch.randn(2, 128, 8, 128, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, q, q, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), q.transpose(1, 2), q.transpose(1, 2), is_causal=True).transpose(1, 2)
    print("flash_attn max abs diff vs sdpa:", (out - ref).abs().max().item())

    llm = LLM("Qwen/Qwen3-0.6B", gpu_memory_utilization=0.3, max_model_len=2048)
    o = llm.generate(["Write a Python function that reverses a string."],
                     SamplingParams(max_tokens=64, temperature=0))
    print("vllm sample:", repr(o[0].outputs[0].text[:200]))


if __name__ == "__main__":
    main()
