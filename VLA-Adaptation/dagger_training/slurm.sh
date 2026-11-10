#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --requeue
set -euo pipefail
: "${OPENPI_ROOT:?Set OPENPI_ROOT to the patched OpenPI cooking checkout}"
: "${OPENPI_PYTHON:?Set OPENPI_PYTHON to its environment Python executable}"
# Slurm executes a spool copy of this script; resolve the project from submission.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  TASK_DIR="${DAGGER_TRAINING_DIR:-${SLURM_SUBMIT_DIR:?}/dagger_training}"
else
  TASK_DIR="${DAGGER_TRAINING_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
fi
[[ -f "$TASK_DIR/dagger-1/train_dagger.py" ]] || {
  echo "Training directory missing: $TASK_DIR; set DAGGER_TRAINING_DIR" >&2; exit 2;
}
unset JAX_PLATFORMS
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
case "${1:-union}" in
  dagger) exec "$OPENPI_PYTHON" "$TASK_DIR/dagger-1/train_dagger.py" ;;
  union) exec "$OPENPI_PYTHON" "$TASK_DIR/union-comparison/train_union.py" --mode union ;;
  dagger-eval) exec "$OPENPI_PYTHON" "$TASK_DIR/union-comparison/train_union.py" --mode dagger_eval ;;
  *) echo 'Choose dagger, union, or dagger-eval' >&2; exit 2 ;;
esac
