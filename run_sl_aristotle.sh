#!/bin/bash
#SBATCH --job-name=chess-sl-mask
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module load gcc/14.2.0

cd $HOME/chess-agent-aristotle
source ~/pytorch-env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${SCRATCH:?SCRATCH is not set}"

export SL_SHARD_DIR="$SCRATCH/chess-agent-aristotle/outputs/sl_shards"
mkdir -p "$SL_SHARD_DIR"

echo "Using SL_SHARD_DIR=$SL_SHARD_DIR"

python -u chess-agent.py train_sl \
  --pgn $HOME/chess-agent-aristotle/data/lichess_elite_2024-07.pgn \
  --sl-epochs 6 \
  --sl-value-min-ply 10 \
  --sl-value-positions-per-game 5 