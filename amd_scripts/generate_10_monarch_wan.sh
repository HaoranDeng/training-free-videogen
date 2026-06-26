#!/bin/bash

#SBATCH -J TFVGen
#SBATCH -o out/job.%j.out
#SBATCH -e out/job.%j.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 4:00:00
#SBATCH -p mi3001x

set -euo pipefail

if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [ -n "${WORK:-}" ] && [ -f "${WORK}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${WORK}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

conda activate monarch_rt

REPO_DIR="${REPO_DIR:-/home1/denghaoran/workspace/training-free-videogen}"
export WAN_MODEL_ROOT="${WAN_MODEL_ROOT:-/home1/denghaoran/workspace/MonarchRT/wan_models}"
CHECKPOINT="${CHECKPOINT:-/work1/jasoncong/denghaoran/MonarchRT/checkpoints/self_forcing_dmd.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-videos/monarch_wan}"

cd "${REPO_DIR}"
mkdir -p out "${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -mindepth 1 -delete

MIOPEN_CACHE_ROOT="${TMPDIR:-${PWD}/assets}/miopen-${SLURM_JOB_ID:-manual}"
export MIOPEN_USER_DB_PATH="${MIOPEN_CACHE_ROOT}/user-db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CACHE_ROOT}/kernel-cache"
mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python generate.py \
  --config configs/monarch_wan_fewstep.yaml \
  --checkpoint "${CHECKPOINT}" \
  --prompt_path prompts/MovieGenVideoBench_extended.txt \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos 10 \
  --use_ema

