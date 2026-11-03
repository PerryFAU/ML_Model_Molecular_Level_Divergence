#!/bin/bash
# install_pixi.sh
# Location : $HOME/PiXi/install_pixi.sh
#
# PURPOSE
#   Bootstrap Miniconda (if needed) and the PiXi_env conda environment for the
#   PiXi CPU pipeline. Installs Python 3.10, R 4.3, TensorFlow 2.16.1 (Keras 3),
#   and R packages used by pixi_run, ensemble, and plotting scripts.
#
# CALLED BY
#   • End user (one-time setup or env rebuild)
#   • install_pixi.R (subprocess)
#
# REQUIRED ENVIRONMENT VARIABLES
#   HOME – user home (conda and PiXi paths are under HOME)
#
# OPTIONAL ENVIRONMENT VARIABLES
#   FORCE_FRESH_MINICONDA=1 – delete all of $HOME/miniconda3 and reinstall
#                             (default: keep existing Miniconda; only recreate
#                             the PiXi_env environment)
#
# COMMAND-LINE ARGUMENTS
#   (none)
#
# OUTPUT
#   $HOME/miniconda3                 – Miniconda root (created if missing)
#   $HOME/miniconda3/envs/PiXi_env   – env used by run_cpu.sbatch / run_cpu.R /
#                                      ensemble launchers
#
# Usage:
#   cd ~/PiXi
#   bash install_pixi.sh
#   FORCE_FRESH_MINICONDA=1 bash install_pixi.sh   # wipe entire miniconda3
# Example:
#   bash "$HOME/PiXi/install_pixi.sh"
#
# Notes:
#   • Default does NOT wipe other conda envs. Only FORCE_FRESH_MINICONDA=1 does.
#   • TF 2.16.1 is the Keras 3 minimum backend (keras3 R depends on TF >= 2.16).
#     Do not bump TF_VERSION without testing save/load and training.
#   • CPU only. GPU/CUDA belongs in a separate install_pixi_gpu.sh recipe.
#   • Needs outbound network (Anaconda, PyPI, CRAN, Bioconductor) on the
#     install node; use mirrors or offline wheels if the cluster blocks them.
#   • Base R stack from conda-forge; keras3 / scoup / extras from CRAN/Bioc.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------
# BLOCK 1 — Variables
# ---------------------------------------------------------
PIXIDIR="${HOME}/PiXi"
CONDA_ROOT="${HOME}/miniconda3"
ENV_NAME="PiXi_env"
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"

PYTHON_VERSION="3.10"
R_BASE_VERSION="4.3"
MINICONDA_INSTALLER="Miniconda3-py310_24.3.0-0-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"

# Pins for reproducible PiXi builds (keras3 requires TF >= 2.16.1)
TF_VERSION="2.16.1"
KERAS_VERSION="3.5.0"

CRAN_REPO="https://cloud.r-project.org"
FORCE_FRESH_MINICONDA="${FORCE_FRESH_MINICONDA:-0}"

# ---------------------------------------------------------
# BLOCK 2 — Preconditions
# ---------------------------------------------------------
if [ -z "${HOME:-}" ]; then
  echo "ERROR: HOME is not set." >&2
  exit 1
fi
if [ ! -d "${PIXIDIR}" ]; then
  echo "ERROR: PiXi directory not found: ${PIXIDIR}" >&2
  exit 1
fi

cd "${PIXIDIR}"
echo "=== PiXi environment install ==="
echo "  PIXIDIR                = ${PIXIDIR}"
echo "  CONDA_ROOT             = ${CONDA_ROOT}"
echo "  ENV_PREFIX             = ${ENV_PREFIX}"
echo "  TF_VERSION             = ${TF_VERSION}"
echo "  KERAS_VERSION          = ${KERAS_VERSION}"
echo "  FORCE_FRESH_MINICONDA  = ${FORCE_FRESH_MINICONDA}"
echo ""

# ---------------------------------------------------------
# BLOCK 3 — Miniconda (install or reuse)
# ---------------------------------------------------------
# Full wipe only when FORCE_FRESH_MINICONDA=1. Otherwise keep existing
# Miniconda and only replace PiXi_env later.

if [ "${FORCE_FRESH_MINICONDA}" = "1" ]; then
  echo "--- FORCE_FRESH_MINICONDA=1: removing ${CONDA_ROOT} ---"
  rm -rf "${CONDA_ROOT}"
fi

if [ ! -x "${CONDA_ROOT}/bin/conda" ]; then
  echo "--- Installing Miniconda under ${CONDA_ROOT} ---"
  mkdir -p "${CONDA_ROOT}"
  wget "${MINICONDA_URL}" -O "${CONDA_ROOT}/miniconda.sh"
  bash "${CONDA_ROOT}/miniconda.sh" -b -u -p "${CONDA_ROOT}"
  rm -f "${CONDA_ROOT}/miniconda.sh"
else
  echo "--- Reusing existing Miniconda at ${CONDA_ROOT} ---"
fi

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "Miniconda OK: $(conda --version)"
echo ""

# ---------------------------------------------------------
# BLOCK 4 — Create or Recreate PiXi_env
# ---------------------------------------------------------
# Remove only this env so other projects under the same Miniconda survive.

if [ -d "${ENV_PREFIX}" ]; then
  echo "--- Removing existing env ${ENV_NAME} ---"
  conda env remove -y -p "${ENV_PREFIX}" || rm -rf "${ENV_PREFIX}"
fi

echo "--- Creating conda env ${ENV_NAME} (conda-forge base R stack) ---"
conda create -y -p "${ENV_PREFIX}" \
  "python=${PYTHON_VERSION}" \
  "r-base=${R_BASE_VERSION}" \
  "r-reticulate>=1.41" \
  r-dplyr r-readr r-tidyr r-moments \
  r-ggplot2 r-viridis r-pROC r-png \
  compilers zlib \
  -c conda-forge

conda activate "${ENV_PREFIX}"
export R_LIBS_USER=""

echo "Activated: ${CONDA_PREFIX:-${ENV_PREFIX}}"
echo "Python: $(command -v python)"
echo "R:      $(command -v R)"
echo ""

# ---------------------------------------------------------
# BLOCK 5 — Python TensorFlow / Keras 3
# ---------------------------------------------------------
echo "--- pip: TensorFlow ${TF_VERSION} + Keras ${KERAS_VERSION} ---"
python -m pip install --upgrade pip
python -m pip install "tensorflow==${TF_VERSION}" "keras==${KERAS_VERSION}"
echo ""

# ---------------------------------------------------------
# BLOCK 6 — R Packages (CRAN / Bioc into conda library)
# ---------------------------------------------------------
# keras3 + tensorflow R bindings and packages not taken from conda-forge.
# Optional keras3 helpers: tfruns, callr, withr.

echo "--- R packages into conda library (CRAN + Bioconductor) ---"
Rscript -e "
  options(repos = c(CRAN = '${CRAN_REPO}'))
  pkgs <- c(
    'tensorflow', 'keras', 'keras3',
    'Matrix', 'mvtnorm', 'seqinr', 'ape',
    'BiocManager', 'ggsignif', 'jsonlite',
    'tfruns', 'callr', 'withr'
  )
  install.packages(pkgs, dependencies = TRUE)
  if (!requireNamespace('BiocManager', quietly = TRUE))
    stop('BiocManager failed to install')
  BiocManager::install('scoup', update = FALSE, ask = FALSE)
"
echo ""

# ---------------------------------------------------------
# BLOCK 7 — Verification
# ---------------------------------------------------------
echo "--- Verification ---"
Rscript -e '
  cat("R version      :", R.version.string, "\n")
  cat("reticulate     :", as.character(packageVersion("reticulate")), "\n")
  cat("keras3         :", as.character(packageVersion("keras3")), "\n")
  cat("tensorflow (R) :", as.character(packageVersion("tensorflow")), "\n")
  library(reticulate)
  Sys.setenv(
    RETICULATE_PYTHON = file.path(Sys.getenv("CONDA_PREFIX"), "bin", "python")
  )
  cat("Python         :", py_config()$python, "\n")
  tf <- import("tensorflow")
  cat("TF (Python)    :", as.character(tf$`__version__`), "\n")
  k <- import("keras")
  cat("Keras (Python) :", as.character(k$`__version__`), "\n")
  cat("OK\n")
'

echo ""
echo "Installed: ${ENV_PREFIX}"
echo "Check with: ${ENV_PREFIX}/bin/Rscript ${PIXIDIR}/install_pixi.R check"
echo "=== PiXi environment install complete ==="