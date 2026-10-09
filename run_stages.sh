#!/bin/bash
# Train the two arms on BABILong with the three-stage curriculum of train_stages.py:
# the vanilla transformer with the CCCCFFFF encoder mask (no positional encoding), and ALiBi.
#
#   bash run_stages.sh              # one Slurm job per model
#   bash run_stages.sh --local      # run here, one after another
#   bash run_stages.sh --dry-run    # list what would run
#
# Training lengths are 0-2k; after every stage each model is also evaluated at 4k and 8k.
# Extra flags for train_stages.py go in EXTRA, e.g. EXTRA="--epochs 2 2 2", or
# EXTRA="--train_len 0 0 --eval_lens 0" for the bare facts with no PG19 download.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
EXP_ROOT="${EXP_ROOT:-${REPO_DIR}/experiments}"
LOG_DIR="${EXP_ROOT}/slurm_logs"
SEED="${SEED:-42}"
EXTRA="${EXTRA:-}"

PARTITION="${PARTITION:-gpu}"
GPU_SPEC="${GPU_SPEC:-h100-96:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-48:00:00}"

MODE="${1:-slurm}"

# name | model flags
RUNS=(
    "transformer_mask_penone_encCCCCFFFF_s${SEED}|--model transformer_mask --encoder_mask CCCCFFFF"
    "alibi_s${SEED}|--model alibi"
)

train_run() {
    local name="$1" flags="$2"
    export PATH="${VENV_DIR}/bin:${PATH}"
    cd "${REPO_DIR}"
    [ -d data/tasks_1-20_v1-2 ] || unzip -q data/tasks_1-20_v1-2.zip -d data
    echo "--- ${name} ---"
    python train_stages.py ${flags} --seed "${SEED}" --output_dir "${EXP_ROOT}/${name}" ${EXTRA}
}

mkdir -p "${LOG_DIR}"
for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    flags="${entry#*|}"
    case "${MODE}" in
        --dry-run) echo "[dry-run] ${name}: train_stages.py ${flags} --seed ${SEED} ${EXTRA}" ;;
        --local)   train_run "${name}" "${flags}" ;;
        slurm)
            sbatch --job-name="babilong_${name}" --partition="${PARTITION}" --gpus="${GPU_SPEC}" \
                   --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}" \
                   --output="${LOG_DIR}/${name}_%j.out" \
                   --error="${LOG_DIR}/${name}_%j.err" <<SBATCH
#!/bin/bash
set -euo pipefail
$(declare -p REPO_DIR VENV_DIR EXP_ROOT SEED EXTRA)
$(declare -f train_run)
train_run "${name}" "${flags}"
SBATCH
            echo "  -> submitted babilong_${name}" ;;
        *) echo "ERROR: unknown argument '${MODE}' (expected --local or --dry-run)." >&2; exit 1 ;;
    esac
done
