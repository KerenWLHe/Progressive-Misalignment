#!/bin/bash -l
# Build flash-attn 2.8.3 from source for sm_120 inside "weilei-sdpo". The prebuilt wheel
# needs GLIBC 2.32 but BCC nodes have 2.28; the system gcc (8.5) and CUDA (13.0) do not
# match torch 2.8+cu128, so nvcc 12.8 and gcc 12 come from conda. CPU-only job:
#   sbatch -p rai -c 32 --mem=128G -t 02:00:00 -J build_fa -o logs/%x-%j.out scripts/build_flash_attn.sh
set -eo pipefail
source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh
conda activate weilei-sdpo
export PIP_CACHE_DIR=/data/ZhouxingShi/whe063/.cache/pip

conda list cuda-nvcc | grep -q 12.8 || conda install -y -c nvidia/label/cuda-12.8.1 -c conda-forge \
    cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-libraries-dev=12.8 "gxx_linux-64=12" "gcc_linux-64=12"
conda deactivate && conda activate weilei-sdpo  # load the conda gcc activation (CC/CXX)
export CUDA_HOME=$CONDA_PREFIX
# rai-navel's root filesystem (and so /tmp) is full; nvcc needs scratch space
export TMPDIR=/data/ZhouxingShi/whe063/tmp && mkdir -p $TMPDIR
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=16 NVCC_THREADS=2
nvcc --version | tail -2

pip uninstall -y flash-attn
pip install ninja packaging psutil
FULL=/data/ZhouxingShi/whe063/projects/progressive-misalignment/logs/fa_build_full.log
pip install flash-attn==2.8.3 --no-build-isolation --no-binary flash-attn --no-cache-dir -v > $FULL 2>&1 ||
    { grep -m 30 -B2 -E "error|fatal" $FULL; exit 1; }
python -c "import flash_attn, flash_attn_2_cuda; print('flash_attn', flash_attn.__version__, 'OK')"
echo BUILD_DONE
