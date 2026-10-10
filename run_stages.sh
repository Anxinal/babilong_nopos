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
#
# Each model is one wandb run (train loss per step, validation per epoch, accuracy by
# length per stage). Jobs need credentials on the compute nodes: `wandb login` on a
# shared home or WANDB_API_KEY. WANDB_MODE=offline logs locally for a later `wandb sync`;
# disabled turns it off.
#
# The Python environment is set up by the job itself: the virtualenv at VENV_DIR is
# created on first use and filled from requirements-train.txt, and the bAbI archive is
# unpacked. Nothing has to be installed by hand first.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Python environment. The virtualenv must live somewhere every compute node can see,
# which the repo root normally is; override VENV_DIR if your home is not shared.
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
REQUIREMENTS="${REQUIREMENTS:-${REPO_DIR}/requirements-train.txt}"
# Interpreter used to build the venv, not the one inside it.
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
# Extra flags for every pip install, for sites where PyPI is not directly reachable
# from a compute node, e.g. PIP_ARGS='--index-url https://<internal-mirror>/simple'.
PIP_ARGS="${PIP_ARGS:-}"
# A plain PyPI torch wheel may be CPU-only or built against the wrong CUDA. Name the
# exact wheel for this cluster and it is installed first, e.g.
#   TORCH_SPEC='torch --index-url https://download.pytorch.org/whl/cu121'
TORCH_SPEC="${TORCH_SPEC:-}"
EXP_ROOT="${EXP_ROOT:-${REPO_DIR}/experiments}"
LOG_DIR="${EXP_ROOT}/slurm_logs"
SEED="${SEED:-42}"
EXTRA="${EXTRA:-}"

WANDB_PROJECT="${WANDB_PROJECT:-Babilong_nopos}"
WANDB_ENTITY="${WANDB_ENTITY:-}"                 # empty: your default entity
WANDB_GROUP="${WANDB_GROUP:-stages}"
WANDB_MODE="${WANDB_MODE:-online}"               # online | offline | disabled

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

# Create the virtualenv if it is missing, install the requirements if they changed, and
# put it first on PATH. Activation is done by hand rather than by sourcing bin/activate,
# because that script touches unset variables and this runs under 'set -u'.
setup_env() {
    # The two jobs start together and share one virtualenv: the first builds it, the
    # other waits here. (No flock on macOS; --local runs one after another anyway.)
    if command -v flock >/dev/null 2>&1; then
        exec 9>"${VENV_DIR}.lock"
        flock 9
    fi

    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        echo "--- Creating virtualenv: ${VENV_DIR} ---"
        if ! "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"; then
            echo "ERROR: could not create a virtualenv with '${BOOTSTRAP_PYTHON} -m venv'." >&2
            echo "       Point BOOTSTRAP_PYTHON at a usable interpreter, or load a python" >&2
            echo "       module first. Some sites need the python3-venv package." >&2
            exit 1
        fi
    fi
    export VIRTUAL_ENV="${VENV_DIR}"
    export PATH="${VENV_DIR}/bin:${PATH}"
    unset PYTHONHOME 2>/dev/null || true
    echo "--- Using virtualenv: ${VENV_DIR} ---"

    # Reinstall when the requirements file is newer than the last successful install.
    # The marker is written only on success, so an interrupted install is retried.
    local marker="${VENV_DIR}/.requirements-installed"
    if [ ! -f "${marker}" ] || [ "${REQUIREMENTS}" -nt "${marker}" ]; then
        if [ ! -f "${REQUIREMENTS}" ]; then
            echo "ERROR: requirements file not found: ${REQUIREMENTS}" >&2
            exit 1
        fi
        python -m pip install --quiet --upgrade pip || true
        # A CUDA-specific torch goes in first, so the 'torch>=2.1' line in the
        # requirements is already satisfied and does not pull another wheel over it.
        if [ -n "${TORCH_SPEC}" ] && ! python -c "import torch" 2>/dev/null; then
            echo "--- Installing torch from TORCH_SPEC: ${TORCH_SPEC} ---"
            # shellcheck disable=SC2086
            pip_install ${TORCH_SPEC}
        fi
        echo "--- Installing from ${REQUIREMENTS} ---"
        pip_install -r "${REQUIREMENTS}"
        touch "${marker}"
    fi
    python -c "import sys, torch, transformers, datasets, wandb
print('    python', sys.version.split()[0], '| torch', torch.__version__, '| transformers',
      transformers.__version__, '| datasets', datasets.__version__, '| cuda', torch.cuda.is_available())"

    cd "${REPO_DIR}"
    [ -d data/tasks_1-20_v1-2 ] || unzip -q data/tasks_1-20_v1-2.zip -d data
    if command -v flock >/dev/null 2>&1; then
        flock -u 9
    fi
}

# `python -m pip` rather than a bare `pip`, which on a cluster often resolves to a
# different environment than `python` does.
pip_install() {
    # shellcheck disable=SC2086
    if ! python -m pip install --no-input --disable-pip-version-check ${PIP_ARGS} "$@"; then
        echo "ERROR: pip install failed. The usual cause is that this compute node has no" >&2
        echo "       outbound network; point PIP_ARGS at a reachable source, e.g." >&2
        echo "           PIP_ARGS='--index-url https://<internal-mirror>/simple' bash run_stages.sh" >&2
        exit 1
    fi
}

train_run() {
    local name="$1" flags="$2"
    setup_env
    echo "--- ${name} ---"
    python train_stages.py ${flags} --seed "${SEED}" --output_dir "${EXP_ROOT}/${name}" \
        --project "${WANDB_PROJECT}" --group "${WANDB_GROUP}" --mode "${WANDB_MODE}" \
        ${WANDB_ENTITY:+--entity "${WANDB_ENTITY}"} ${EXTRA}
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
$(declare -p REPO_DIR VENV_DIR REQUIREMENTS BOOTSTRAP_PYTHON PIP_ARGS TORCH_SPEC \
             EXP_ROOT SEED EXTRA WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_MODE)
$(declare -f setup_env pip_install train_run)
# A job without a GPU would train on the CPU until the time limit; stop instead.
setup_env
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'ERROR: torch sees no GPU in this job; set TORCH_SPEC to a CUDA build.')"
train_run "${name}" "${flags}"
SBATCH
            echo "  -> submitted babilong_${name}" ;;
        *) echo "ERROR: unknown argument '${MODE}' (expected --local or --dry-run)." >&2; exit 1 ;;
    esac
done
