# shellcheck shell=bash
# Shared PBS environment setup. Source from a PBS script:
#     source "${REPO}/scripts/pbs_env.sh"
# Sets PYTHON (honouring PYTHON_BIN / PYTHON if already set, else activating
# mamba env MAMBA_ENV_NAME, default "astro") and caps BLAS/OpenMP threads.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
unset PYTHONHOME || true
unset PYTHONPATH || true

MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-astro}"

activate_mamba() {
    if [[ -n "${MAMBA_EXE:-}" && -x "${MAMBA_EXE}" ]]; then
        eval "$("${MAMBA_EXE}" shell hook --shell bash)"
    elif command -v mamba >/dev/null 2>&1; then
        eval "$(mamba shell hook --shell bash)"
    elif [[ -f "${HOME}/miniforge/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniforge/etc/profile.d/conda.sh"
    else
        return 1
    fi
    if command -v mamba >/dev/null 2>&1; then
        mamba activate "${MAMBA_ENV_NAME}"
    elif command -v conda >/dev/null 2>&1; then
        conda activate "${MAMBA_ENV_NAME}"
    else
        return 1
    fi
}

PYTHON="${PYTHON_BIN:-${PYTHON:-}}"
if [[ -z "${PYTHON}" ]]; then
    if ! activate_mamba; then
        echo "[pbs_env] WARN: could not activate mamba env '${MAMBA_ENV_NAME}', falling back to PATH python." >&2
    fi
    PYTHON="$(command -v python || command -v python3 || true)"
    if [[ -z "${PYTHON}" ]]; then
        echo "[pbs_env] ERROR: no python executable found after env setup." >&2
        exit 1
    fi
fi
export PYTHON

to_bool01() {
    local val
    val="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
    case "${val}" in
        1|true|yes|on) echo 1 ;;
        *) echo 0 ;;
    esac
}
