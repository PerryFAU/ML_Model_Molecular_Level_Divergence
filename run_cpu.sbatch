#!/bin/bash
# run_cpu.sbatch
# Location : $HOME/PiXi/run_cpu.sbatch
#
# PURPOSE
#   Slurm batch launcher for a PiXi CPU run. Allocates a long-queue CPU node,
#   activates the PiXi conda environment, and runs pixi_run.R with MODE=CPU.
#
# CALLED BY
#   • End user via: sbatch run_cpu.sbatch <run_dir>
#
# REQUIRED ENVIRONMENT / PATHS
#   HOME must point to the user home directory.
#   Expects conda env PiXi_env and pixi_run.R under $HOME/PiXi.
#
# COMMAND-LINE ARGUMENTS
#   run_dir – name of the run directory to create under $HOME/PiXi
#
# OUTPUT
#   Creates $HOME/PiXi/<run_dir> (via pixi_run.R) and writes pipeline artifacts.
#   After the job finishes, relocates Slurm stdout/stderr into that run dir
#   (filenames keep SLURM_JOB_ID so re-runs do not overwrite prior logs).
#
# Usage:
#   sbatch run_cpu.sbatch <run_dir>
# Example:
#   sbatch run_cpu.sbatch Test_Run_001
#
# Related:
#   • Interactive CPU twin: run_cpu.R (same PIXI_ROOT / conda / MODE ideas)
#   • GPU batch twin: run_gpu.sbatch (mirror this layout with GPU partition)
#
# ---------------------------------------------------------------------------
# ABOUT THIS FILE (Slurm)
# ---------------------------------------------------------------------------
# This is a Slurm batch script. It is *not* executed directly by the shell.
# Submit with:  sbatch run_cpu.sbatch <run_dir>
#
# When submitted, Slurm:
#   1. Reads this file as a job description.
#   2. Parses all lines beginning with #SBATCH to configure the job.
#   3. Allocates the requested resources (nodes, CPUs, memory, time, etc.).
#   4. Starts a new job allocation on a compute node.
#   5. Launches bash inside the allocated job environment.
#   6. Executes the remaining lines of this file as a shell script.
#
# Notes:
#   • The file does NOT need to be executable (chmod +x has no effect).
#   • The shebang does not control how Slurm launches the job.
#   • Log relocation must stay at the END (see BLOCK 6) — moving logs early
#     can cause Slurm to re-execute the script and double-run pixi_run.R.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------
# BLOCK 0 — Slurm Job Configuration
# ---------------------------------------------------------
# Resource request for a single-node CPU job. #SBATCH lines are parsed by
# “Slurm before any shell variables expand; keep literals here.
#
# Smoke-test alternative (comment long settings, uncomment short):
#   partition=shortq7   time=06:00:00
# Production default:
#   partition=longq7    time=7-00:00:00

#SBATCH --job-name=pixi_cpu
# #SBATCH --partition=shortq7
#SBATCH --partition=longq7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000M
# #SBATCH --time=06:00:00
#SBATCH --time=7-00:00:00
#SBATCH --output=pixi_cpu_%j.out
#SBATCH --error=pixi_cpu_%j.err

# ---------------------------------------------------------
# BLOCK 1 — Variables
# ---------------------------------------------------------
# Paths and mode (keep in sync with run_cpu.R where practical).

PIXI_ROOT="${HOME}/PiXi"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${HOME}/miniconda3/envs/PiXi_env"
RSCRIPT_BIN="${CONDA_ENV}/bin/Rscript"
PYTHON_BIN="${CONDA_ENV}/bin/python"
PIXI_RUN_SCRIPT="${PIXI_ROOT}/pixi_run.R"
MODE_ARG="CPU"
DEFAULT_MODE="cpu"

# ---------------------------------------------------------
# BLOCK 2 — Export Mode and Working Directory
# ---------------------------------------------------------
# Require run_dir, export PIXI_* for pixi_run.R, create the run root.

run_dir="$1"
if [ -z "$run_dir" ]; then
  echo "ERROR: You must supply a run directory name." >&2
  echo "Usage: sbatch run_cpu.sbatch <run_dir>" >&2
  exit 1
fi

export PIXI_MODE="${DEFAULT_MODE}"
export PIXI_WORKDIR="${PIXI_ROOT}/${run_dir}"
mkdir -p "${PIXI_WORKDIR}"

echo "PiXi CPU job starting"
echo "  PIXI_MODE     = ${PIXI_MODE}"
echo "  PIXI_WORKDIR  = ${PIXI_WORKDIR}"
echo "  SLURM_JOB_ID  = ${SLURM_JOB_ID:-none}"
echo ""

# ---------------------------------------------------------
# BLOCK 3 — Activate CPU Environment
# ---------------------------------------------------------
# Load conda, activate PiXi_env, force reticulate Python, clear personal R libs.

# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export RETICULATE_PYTHON="${PYTHON_BIN}"
export R_LIBS_USER=""

echo "  RETICULATE_PYTHON = ${RETICULATE_PYTHON}"
echo "  CONDA_PREFIX      = ${CONDA_PREFIX:-unset}"
echo ""

# ---------------------------------------------------------
# BLOCK 4 — Working Directory for Driver
# ---------------------------------------------------------
# Run from PiXi root so any relative paths inside the driver resolve consistently.

cd "${PIXI_ROOT}" || {
  echo "ERROR: Cannot cd to ${PIXI_ROOT}" >&2
  exit 1
}
echo "  cwd = $(pwd)"
echo ""

# ---------------------------------------------------------
# BLOCK 5 — Execute PiXi Driver Script
# ---------------------------------------------------------
# Confirm R home, then run pixi_run.R with MODE=CPU as a subprocess.

"${RSCRIPT_BIN}" -e "cat('R HOME = ', R.home(), '\n')"

if [ ! -f "${PIXI_RUN_SCRIPT}" ]; then
  echo "ERROR: Cannot find pixi_run.R at: ${PIXI_RUN_SCRIPT}" >&2
  exit 1
fi

echo "Launching: ${RSCRIPT_BIN} ${PIXI_RUN_SCRIPT} ${MODE_ARG}"
"${RSCRIPT_BIN}" "${PIXI_RUN_SCRIPT}" "${MODE_ARG}"
status=$?
if [ "${status}" -ne 0 ]; then
  echo "ERROR: pixi_run.R exited with status ${status}" >&2
  exit "${status}"
fi

# Plotting is invoked from inside pixi_run.R (not a separate sbatch here).

# ---------------------------------------------------------
# BLOCK 6 — Move Slurm Logs
# ---------------------------------------------------------
# Relocate stdout/stderr into the run directory only after the driver finishes.
# Keep SLURM_JOB_ID in the filename so re-runs in the same run_dir do not
# overwrite earlier logs.

if [ -n "${SLURM_JOB_ID}" ]; then
  if [ -f "pixi_cpu_${SLURM_JOB_ID}.out" ]; then
    mv "pixi_cpu_${SLURM_JOB_ID}.out" \
      "${PIXI_WORKDIR}/pixi_cpu_${SLURM_JOB_ID}.out"
    echo "Moved stdout log -> ${PIXI_WORKDIR}/pixi_cpu_${SLURM_JOB_ID}.out"
  fi
  if [ -f "pixi_cpu_${SLURM_JOB_ID}.err" ]; then
    mv "pixi_cpu_${SLURM_JOB_ID}.err" \
      "${PIXI_WORKDIR}/pixi_cpu_${SLURM_JOB_ID}.err"
    echo "Moved stderr log -> ${PIXI_WORKDIR}/pixi_cpu_${SLURM_JOB_ID}.err"
  fi
fi

echo "PiXi CPU job finished."
exit 0