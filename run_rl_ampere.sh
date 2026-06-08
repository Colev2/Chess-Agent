#!/bin/bash
#SBATCH --job-name=chess-rl
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module load gcc/14.2.0

cd $HOME/chess-agent-aristotle
source ~/pytorch-env/bin/activate

: "${SCRATCH:?SCRATCH is not set}"

export CHESS_SCRATCH_OUTPUT_DIR="$SCRATCH/chess-agent-aristotle/outputs"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


python -u chess_agent.py train_rl --fresh-rl-from-sl
