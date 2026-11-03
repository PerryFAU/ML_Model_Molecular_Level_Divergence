# ensemble_run.R
# Location : $HOME/PiXi/ensemble_run.R
#
# PURPOSE
#   Unified ensemble driver. Binds the CPU/GPU environment, sources ensemble
#   modules, builds fused training/testing data from expression/sequence/
#   structure base-model runs, trains the ensemble classifier and predictor,
#   runs ensemble testing, and launches ensemble plotting.
#
# CALLED BY
#   • run_ensemble_cpu.R (interactive launcher)
#   • run_ensemble_cpu.sbatch (Slurm)
#
# REQUIRED ENVIRONMENT VARIABLES
#   PIXI_WORKDIR            – base run directory for this ensemble job
#   PIXI_MODE               – "cpu" or "gpu" (default: "cpu"; may be set from CLI)
#   ENSEMBLE_EXPR_DIR       – base-model expression Training parent
#   ENSEMBLE_SEQ_DIR        – base-model sequence Training parent
#   ENSEMBLE_STR_DIR        – base-model structure Training parent
#   ENSEMBLE_EXPR_TEST_DIR  – base-model expression Testing parent
#   ENSEMBLE_SEQ_TEST_DIR   – base-model sequence Testing parent
#   ENSEMBLE_STR_TEST_DIR   – base-model structure Testing parent
#
# COMMAND-LINE ARGUMENTS
#   MODE – optional first argument ("cpu" or "gpu"). If provided, PIXI_MODE
#          is set from it so paths and Python binding stay consistent.
#
# OUTPUT
#   Under PIXI_WORKDIR/<CPU|GPU>/:
#     Training/Ensemble_Fused.*  – fused data, models, CV summaries
#     Testing/Ensemble_Fused.*   – fused test features and predictions
#     metrics/, plots/           – ensemble metrics and figures
#
# Usage:
#   Rscript ensemble_run.R
#   Rscript ensemble_run.R cpu
#   Rscript ensemble_run.R gpu
#
# ---------------------------------------------------------
# BLOCK 0 — Variables
# ---------------------------------------------------------
# Tunable depths, training hyperparams, and path roots.

# Ensemble network depths (0 = linear / no hidden layers)
num_layers_ensemble_cls  <- 0
num_layers_ensemble_pred <- 0

# Base-model depths used when building fused features (must match those runs)
num_layers_expr   <- 2
num_layers_seq    <- 2
num_layers_struct <- 2

# Shared CV / training hyperparams for ensemble classifier and predictor
batchsize      <- 10
num_epochs     <- 5
log_lambda_min <- -3
log_lambda_max <- 3
num_lambda     <- 7
gamma_min      <- 0
gamma_max      <- 1
num_gamma      <- 3

# Path roots
pixi_root         <- file.path(Sys.getenv("HOME"), "PiXi")
conda_env_bin     <- file.path(Sys.getenv("HOME"), "miniconda3/envs/PiXi_env/bin")
conda_gpu_env_bin <- file.path(Sys.getenv("HOME"), "miniconda3/envs/PiXi_env/bin")
# ---------------------------------------------------------
# BLOCK 1 — Header Message and Load Libraries
# ---------------------------------------------------------
cat("=== Ensemble run start ===\n")

library(tensorflow)
library(keras)
library(keras3)

# ---------------------------------------------------------
# BLOCK 2 — Resolve Mode (CLI overrides PIXI_MODE)
# ---------------------------------------------------------
# If the first CLI arg is cpu/gpu, set PIXI_MODE from it so directory routing
# and Python binding cannot disagree with the requested mode.
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  mode_arg <- tolower(args[1])
  if (!(mode_arg %in% c("cpu", "gpu"))) {
    stop("MODE must be 'cpu' or 'gpu' (got: ", args[1], ")")
  }
  Sys.setenv(PIXI_MODE = mode_arg)
  cat("PIXI_MODE set from CLI argument:", mode_arg, "\n")
}

pixi_mode <- tolower(Sys.getenv("PIXI_MODE", unset = "cpu"))
if (!(pixi_mode %in% c("cpu", "gpu"))) {
  stop("PIXI_MODE must be 'cpu' or 'gpu' (got: ", pixi_mode, ")")
}
MODE <- toupper(pixi_mode)

cat("PIXI_WORKDIR =", Sys.getenv("PIXI_WORKDIR"), "\n")
cat("PIXI_MODE    =", pixi_mode, "\n")
cat("MODE         =", MODE, "\n")

# ---------------------------------------------------------
# BLOCK 3 — Validate Required Environment Variables
# ---------------------------------------------------------
# Fail early if workdir or base-model paths are missing.
required_env <- c(
  "PIXI_WORKDIR",
  "ENSEMBLE_EXPR_DIR",
  "ENSEMBLE_SEQ_DIR",
  "ENSEMBLE_STR_DIR",
  "ENSEMBLE_EXPR_TEST_DIR",
  "ENSEMBLE_SEQ_TEST_DIR",
  "ENSEMBLE_STR_TEST_DIR"
)

env_errors <- c()
for (nm in required_env) {
  val <- Sys.getenv(nm, unset = "")
  if (!nzchar(val)) {
    env_errors <- c(env_errors, paste0(nm, " is not set"))
  } else if (!dir.exists(val)) {
    env_errors <- c(env_errors, paste0(nm, " does not exist: ", val))
  }
}

if (length(env_errors) > 0) {
  stop(
    "Ensemble environment validation failed:\n  - ",
    paste(env_errors, collapse = "\n  - ")
  )
}

cat("Ensemble env paths OK:\n")
cat("  EXPR train =", Sys.getenv("ENSEMBLE_EXPR_DIR"), "\n")
cat("  SEQ  train =", Sys.getenv("ENSEMBLE_SEQ_DIR"), "\n")
cat("  STR  train =", Sys.getenv("ENSEMBLE_STR_DIR"), "\n")
cat("  EXPR test  =", Sys.getenv("ENSEMBLE_EXPR_TEST_DIR"), "\n")
cat("  SEQ  test  =", Sys.getenv("ENSEMBLE_SEQ_TEST_DIR"), "\n")
cat("  STR  test  =", Sys.getenv("ENSEMBLE_STR_TEST_DIR"), "\n")

# ---------------------------------------------------------
# BLOCK 4 — Python Binding
# ---------------------------------------------------------
# Point TensorFlow/reticulate at the mode-matching conda Python.
if (pixi_mode == "gpu") {
  tensorflow::use_python(file.path(conda_gpu_env_bin, "python"), required = TRUE)
  rscript_bin <- file.path(conda_gpu_env_bin, "Rscript")
} else {
  tensorflow::use_python(file.path(conda_env_bin, "python"), required = TRUE)
  rscript_bin <- file.path(conda_env_bin, "Rscript")
}
cat("Python / Rscript env: PiXi_env\n")
# ---------------------------------------------------------
# BLOCK 5 — Source Ensemble Modules
# ---------------------------------------------------------
source(file.path(pixi_root, "feature_extractor_utils.R"))
source(file.path(pixi_root, "ensemble_training_data_generator.R"))
source(file.path(pixi_root, "ensemble_testing_data_generator.R"))
source(file.path(pixi_root, "EnsembleCVnn.R"))
source(file.path(pixi_root, "EnsemblePredictorCVnn.R"))
source(file.path(pixi_root, "PiXiClassifyEnsemble.R"))
source(file.path(pixi_root, "PiXiPredictEnsemble.R"))

# ---------------------------------------------------------
# BLOCK 6 — Generate Fused Ensemble Training Data
# ---------------------------------------------------------
cat("=== Generating fused ensemble training data ===\n")

mode_dir <- if (pixi_mode == "gpu") "GPU" else "CPU"
ensemble_train_prefix <- file.path(
  Sys.getenv("PIXI_WORKDIR"), mode_dir, "Training", "Ensemble_Fused"
)

GenerateFusedEnsembleTrainingData(
  training_prefix_expr   = file.path(Sys.getenv("ENSEMBLE_EXPR_DIR"), "Training"),
  training_prefix_seq    = file.path(Sys.getenv("ENSEMBLE_SEQ_DIR"), "Training"),
  training_prefix_struct = file.path(Sys.getenv("ENSEMBLE_STR_DIR"), "Training"),
  ensemble_prefix        = ensemble_train_prefix,
  num_layers_expr        = num_layers_expr,
  num_layers_seq         = num_layers_seq,
  num_layers_struct      = num_layers_struct
)

cat("=== Fused ensemble training data created ===\n")
cat("  ensemble_train_prefix =", ensemble_train_prefix, "\n")

# ---------------------------------------------------------
# BLOCK 7 — Generate Fused Ensemble Testing Data
# ---------------------------------------------------------
cat("=== Generating fused ensemble testing data ===\n")

ensemble_test_prefix <- file.path(
  Sys.getenv("PIXI_WORKDIR"), mode_dir, "Testing", "Ensemble_Fused"
)

GenerateFusedEnsembleTestingData(
  training_prefix_expr   = file.path(Sys.getenv("ENSEMBLE_EXPR_DIR"), "Training"),
  training_prefix_seq    = file.path(Sys.getenv("ENSEMBLE_SEQ_DIR"), "Training"),
  training_prefix_struct = file.path(Sys.getenv("ENSEMBLE_STR_DIR"), "Training"),
  testing_prefix_expr    = file.path(Sys.getenv("ENSEMBLE_EXPR_TEST_DIR"), "Testing"),
  testing_prefix_seq     = file.path(Sys.getenv("ENSEMBLE_SEQ_TEST_DIR"), "Testing"),
  testing_prefix_str     = file.path(Sys.getenv("ENSEMBLE_STR_TEST_DIR"), "Testing"),
  ensemble_test_prefix   = ensemble_test_prefix,
  num_layers_expr        = num_layers_expr,
  num_layers_seq         = num_layers_seq,
  num_layers_struct      = num_layers_struct
)

cat("=== Fused ensemble testing data created ===\n")
cat("  ensemble_test_prefix =", ensemble_test_prefix, "\n")

# ---------------------------------------------------------
# BLOCK 8 — Shared Output Paths and Directories
# ---------------------------------------------------------
model_output_prefix <- ensemble_train_prefix
cat("model_output_prefix =", model_output_prefix, "\n")

base_dir <- file.path(Sys.getenv("PIXI_WORKDIR"), mode_dir)
dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(base_dir, "metrics"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(base_dir, "plots"),   recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------
# BLOCK 9 — Train Ensemble Classifier
# ---------------------------------------------------------
cat("=== Training Ensemble Classifier ===\n")

ClassifierCVnn_Ensemble_safe(
  num_layers      = num_layers_ensemble_cls,
  batchsize       = batchsize,
  num_epochs      = num_epochs,
  log_lambda_min  = log_lambda_min,
  log_lambda_max  = log_lambda_max,
  num_lambda      = num_lambda,
  gamma_min       = gamma_min,
  gamma_max       = gamma_max,
  num_gamma       = num_gamma,
  ensemble_prefix = model_output_prefix
)
cat("=== Ensemble classifier training complete ===\n")

# ---------------------------------------------------------
# BLOCK 10 — Ensemble Classification on Testing Data
# ---------------------------------------------------------
cat("=== Running Ensemble Testing Classifier ===\n")


PiXiClassifyEnsemble(
  ensemble_prefix      = model_output_prefix,
  num_layers_ensemble  = num_layers_ensemble_cls,
  ensemble_test_prefix = ensemble_test_prefix   # already defined in BLOCK 7
)
cat("=== Ensemble classification complete ===\n")

# ---------------------------------------------------------
# BLOCK 11 — Train Ensemble Predictor
# ---------------------------------------------------------
cat("=== Training Ensemble Predictor ===\n")

PredictorCVnn_Ensemble_safe(
  num_layers      = num_layers_ensemble_pred,
  batchsize       = batchsize,
  num_epochs      = num_epochs,
  log_lambda_min  = log_lambda_min,
  log_lambda_max  = log_lambda_max,
  num_lambda      = num_lambda,
  gamma_min       = gamma_min,
  gamma_max       = gamma_max,
  num_gamma       = num_gamma,
  ensemble_prefix = model_output_prefix
)
cat("=== Ensemble predictor training complete ===\n")

# ---------------------------------------------------------
# BLOCK 12 — Ensemble Regression on Testing Data
# ---------------------------------------------------------
cat("=== Running Ensemble Testing Predictor ===\n")

PiXiPredictEnsemble(
  ensemble_prefix      = model_output_prefix,
  num_layers_ensemble  = num_layers_ensemble_pred,
  ensemble_test_prefix = ensemble_test_prefix
)
cat("=== Ensemble prediction complete ===\n")

# ---------------------------------------------------------
# BLOCK 13 — Ensemble Plotting
# ---------------------------------------------------------
# Use the same conda env as the active mode for the plotting subprocess.
cat("=== Running Ensemble Plotting ===\n")

Sys.setenv(PIXI_WORKDIR = Sys.getenv("PIXI_WORKDIR"))
Sys.setenv(PIXI_MODE    = MODE)

system2(
  rscript_bin,
  args = c(
    file.path(pixi_root, "plot_ensemble_all.R"),
    MODE,
    as.character(num_layers_ensemble_cls),
    as.character(num_layers_ensemble_pred)
  )
)
cat("=== Ensemble plotting complete ===\n")
cat("=== Ensemble run complete ===\n")