#!/usr/bin/env bash
# Reproduce the Reset & Distill + SAC results on ContinualBench.
#
# Usage:
#   scripts/reproduce.sh [--seeds "0 1 2"] [--num-envs 10] [--steps 1000000]
#                        [--mode continual|random|task] [--no-rnd] [--no-wandb]
#                        [-- extra OmegaConf overrides]
#
# Examples:
#   scripts/reproduce.sh                              # R&D, default task order, seed 0
#   scripts/reproduce.sh --seeds "0 1 2" --no-wandb   # three seeds, no wandb
#   scripts/reproduce.sh --no-rnd                     # naive sequential SAC baseline
#   scripts/reproduce.sh -- cl_reg_coef=0.5           # pass any config override through
set -euo pipefail

SEEDS="0"
NUM_ENVS=10
STEPS=1000000
MODE="continual"
RND=true
WANDB=true
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)    SEEDS="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --steps)    STEPS="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --no-rnd)   RND=false; shift ;;
    --no-wandb) WANDB=false; shift ;;
    --)         shift; EXTRA=("$@"); break ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *)          echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

for seed in $SEEDS; do
  echo "=== seed=${seed} mode=${MODE} rnd=${RND} num_envs=${NUM_ENVS} steps_per_task=${STEPS} ==="
  python -m src.train \
    policy=SAC \
    benchmark_mode="${MODE}" \
    rnd="${RND}" \
    seed="${seed}" \
    num_envs="${NUM_ENVS}" \
    train.steps_per_task="${STEPS}" \
    logging.enable_wandb="${WANDB}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
done
