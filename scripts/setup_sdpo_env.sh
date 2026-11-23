#!/bin/bash -l
# Build the conda env "weilei-sdpo" for lasgroup/SDPO (verl) on rai's Blackwell GPUs.
# torch 2.8.0+cu128 / vLLM 0.11.0 / flash-attn 2.8.3 all ship prebuilt wheels with sm_120,
# so nothing is compiled. Run on the login node with nohup (download-bound, no GPU needed):
#   nohup bash scripts/setup_sdpo_env.sh > logs/setup_sdpo_env.log 2>&1 < /dev/null &
set -eo pipefail
SDPO=/data/ZhouxingShi/whe063/projects/SDPO
FA_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q '^weilei-sdpo ' || conda create -y -n weilei-sdpo python=3.12
conda activate weilei-sdpo
export PIP_CACHE_DIR=/data/ZhouxingShi/whe063/.cache/pip
pip install vllm==0.11.0 torch==2.8.0 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install "$FA_WHL"
# SDPO's pinned core deps, minus anything that would replace the torch/vLLM stack above
grep -vE '^(torch|torchvision|torchaudio|vllm|flash)[=<> ]' "$SDPO/requirements.txt" > /tmp/whe063_sdpo_req.txt
pip install -r /tmp/whe063_sdpo_req.txt
pip install -e "$SDPO" --no-deps
python -c "import torch, vllm, flash_attn, verl, transformers; print('torch', torch.__version__, torch.version.cuda, '| vllm', vllm.__version__, '| flash_attn', flash_attn.__version__, '| transformers', transformers.__version__)"
echo SETUP_DONE
