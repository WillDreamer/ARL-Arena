#!/bin/bash
#SBATCH --job-name=grpo-pilot
#SBATCH --account=torch_pr_926_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Usage:  sbatch pilot.sbatch <variant> [extra args for grpo_clipping_experiment.py]
#   e.g.  sbatch pilot.sbatch token_wide --max_steps 80
#         sbatch --time=00:20:00 pilot.sbatch token_wide --max_steps 5 --eval_steps 5 --n_eval 16 --output_root smoke

set -euo pipefail
VARIANT=${1:?variant required}; shift
cd "${SLURM_SUBMIT_DIR}"
source venv/bin/activate
export HF_HOME=${SCRATCH}/.huggingface TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python grpo_clipping_experiment.py --variant "$VARIANT" "$@"
