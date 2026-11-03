#!/usr/bin/env Rscript
# run_cpu.R
# Location : $HOME/PiXi/run_cpu.R
#
# PURPOSE
#   Interactive (non-Slurm) launcher for a PiXi CPU run.
#   Sets required environment variables and launches pixi_run.R
#   as a clean subprocess.
#
# CALLED BY
#   • End user (command line or interactive R session)
#
# REQUIRED ENVIRONMENT VARIABLES
#   HOME – user home directory (used to locate PiXi and conda)
#
# COMMAND-LINE ARGUMENTS
#   run_dir – name of the run directory to create under $HOME/PiXi
#
# OUTPUT
#   Creates $HOME/PiXi/<run_dir> and runs the full PiXi CPU pipeline
#   via pixi_run.R.
#
# shebang ignored
# Usage:
#   Rscript run_cpu.R <run_dir>
# Example:
#   Rscript run_cpu.R Test_Run_001
#
# shebang provides this option
# chmod +x run_cpu.R
# ./run_cpu.R Test_Run_001
#
# -------------------------------------------------
# How to run interactively from inside R
# -------------------------------------------------
# cd ~/PiXi
# conda activate PiXi_env
# R
#
# Then paste:
#
#   run_dir <- "MyRunName"   # ← change this
#   Sys.setenv(PIXI_MODE = "cpu")
#   Sys.setenv(PIXI_WORKDIR = file.path(Sys.getenv("HOME"), "PiXi", run_dir))
#   Sys.setenv(RETICULATE_PYTHON = file.path(Sys.getenv("HOME"),
#                                            "miniconda3/envs/PiXi_env/bin/python"))
#   Sys.setenv(R_LIBS_USER = "")
#   dir.create(Sys.getenv("PIXI_WORKDIR"), recursive = TRUE, showWarnings = FALSE)
#   system2(
#     file.path(Sys.getenv("HOME"), "miniconda3/envs/PiXi_env/bin/Rscript"),
#     args = c(file.path(Sys.getenv("HOME"), "PiXi", "pixi_run.R"), "CPU")
#   )
#
# ---------------------------------------------------------
# BLOCK 0 — Variables
# ---------------------------------------------------------
# Paths and mode defaults.
pixi_root       <- file.path(Sys.getenv("HOME"), "PiXi")
conda_env_bin   <- file.path(Sys.getenv("HOME"), "miniconda3/envs/PiXi_env/bin")
pixi_run_script <- file.path(pixi_root, "pixi_run.R")
default_mode    <- "cpu"    # lower-case value for PIXI_MODE
mode_arg        <- "CPU"    # upper-case argument passed to pixi_run.R

# ---------------------------------------------------------
# BLOCK 1 — Argument Check
# ---------------------------------------------------------
# Require a run directory name on the command line.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("ERROR: You must supply a run directory name.\n",
       "Usage: Rscript run_cpu.R <run_dir>")
}
run_dir <- args[1]

# ---------------------------------------------------------
# BLOCK 2 — Export Mode and Working Directory
# ---------------------------------------------------------
# Set PIXI_MODE and PIXI_WORKDIR, then create the run directory.
Sys.setenv(PIXI_MODE = default_mode)
workdir <- file.path(pixi_root, run_dir)
Sys.setenv(PIXI_WORKDIR = workdir)

dir.create(workdir, recursive = TRUE, showWarnings = FALSE)

cat("PiXi CPU (interactive) job starting\n")
cat("  PIXI_MODE     =", Sys.getenv("PIXI_MODE"), "\n")
cat("  PIXI_WORKDIR  =", Sys.getenv("PIXI_WORKDIR"), "\n\n")

# ---------------------------------------------------------
# BLOCK 3 — Force Correct Python and R Library Paths
# ---------------------------------------------------------
# Point reticulate/TensorFlow at the PiXi conda Python and
# disable any personal R library so the conda library is used.
Sys.setenv(RETICULATE_PYTHON = file.path(conda_env_bin, "python"))
Sys.setenv(R_LIBS_USER = "")

cat("R HOME =", R.home(), "\n")

# ---------------------------------------------------------
# BLOCK 4 — Execute PiXi Driver Script
# ---------------------------------------------------------
# Launch pixi_run.R as a clean subprocess so MODE is passed correctly.
if (!file.exists(pixi_run_script)) {
  stop("ERROR: Cannot find pixi_run.R at:\n  ", pixi_run_script)
}

system2(
  file.path(conda_env_bin, "Rscript"),
  args = c(pixi_run_script, mode_arg)
)

cat("\nPiXi CPU (interactive) run finished.\n")