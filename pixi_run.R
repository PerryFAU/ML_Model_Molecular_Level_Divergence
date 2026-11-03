# pixi_run.R
# Location : $HOME/PiXi/pixi_run.R
#
# PURPOSE
#   Unified PiXi driver script. Performs environment binding, data generation,
#   classifier/predictor training, testing, feature-extractor extraction,
#   and mode-specific directory routing.
#
# CALLED BY
#   Slurm / batch scripts
#     • benchmark_pixi.sh          (CPU + GPU timing)
#     • run_cpu.sbatch
#     • run_gpu.sbatch
#   Interactive launcher
#     • run_cpu.R
#
# REQUIRED ENVIRONMENT VARIABLES
#   PIXI_WORKDIR   – base run directory (must be set by caller)
#   PIXI_MODE      – "cpu" or "gpu" (default: "cpu")
#
# COMMAND-LINE ARGUMENTS
#   MODE           – first argument; defaults to value of PIXI_MODE (upper-cased)
#
# OUTPUT
#   Populates the directory tree under PIXI_WORKDIR with training data,
#   models, classifications, predictions and plots.
#
# ---------------------------------------------------------
# BLOCK 0 — Variables (easy-to-edit defaults)
# ---------------------------------------------------------
# Network depth
num_layers_cls  <- 2
num_layers_pred <- 2

# File prefixes
training_prefix <- "Training"
testing_prefix  <- "Testing"

# Training-data generation
m                     <- 6
Nk                    <- 400
trait_type            <- "OU"
logAlphaMin           <- 0
logAlphaMax           <- 3
logSigmaSqMin         <- -2
logSigmaSqMax         <- 3
zscore_features       <- FALSE
use_empirical_density <- FALSE
synthetic_pool_mode   <- "legacy"   # "legacy" or "iqr"
IQR_K                 <- 1.5

# Classifier / Predictor training hyperparameters
batchsize      <- 10
num_epochs     <- 5
log_lambda_min <- -3
log_lambda_max <- 3
num_lambda     <- 7
gamma_min      <- 0
gamma_max      <- 1
num_gamma      <- 3

# Paths (derived once)
pixi_root          <- file.path(Sys.getenv("HOME"), "PiXi")
empirical_file     <- file.path(pixi_root, "Empirical", "Testing.features")
conda_env_bin      <- file.path(Sys.getenv("HOME"), "miniconda3/envs/PiXi_env/bin")
conda_gpu_env_bin  <- file.path(Sys.getenv("HOME"), "miniconda3/envs/PiXi_gpu_env/bin")

# ---------------------------------------------------------
# BLOCK 1 — Header Message and Load Libraries
# ---------------------------------------------------------
# Print start banner and load required R packages.
# These packages must already be installed in the active conda environment
# (created by install_pixi.sh).
cat("=== PiXi run start ===\n")

library(tensorflow)
library(keras)
library(keras3)

# ---------------------------------------------------------
# BLOCK 2 — Parameter (Mode)
# ---------------------------------------------------------
# Determine the run mode.
# Priority:
#   1. Command-line argument (if supplied)
#   2. PIXI_MODE environment variable
#   3. Default "cpu"
#
# MODE (upper-case) is passed to plot_all.R.
# pixi_mode (lower-case) is used for directory selection and Python binding.
args <- commandArgs(trailingOnly = TRUE)

pixi_mode <- tolower(Sys.getenv("PIXI_MODE", unset = "cpu"))
if (length(args) >= 1) {
  pixi_mode <- tolower(args[1])
}
MODE <- toupper(pixi_mode)          # "CPU" or "GPU" for plotting

cat("PiXi MODE     =", MODE, "\n")
cat("PIXIRUN: PIXI_MODE (pixi_mode) =", pixi_mode, "\n")

# ---------------------------------------------------------
# BLOCK 3 — Bind Python According to Mode
# ---------------------------------------------------------
# Bind reticulate / TensorFlow to the correct Python interpreter.
# Essential for TensorFlow to locate the proper conda environment.
if (pixi_mode == "gpu") {
    tensorflow::use_python(
        file.path(conda_gpu_env_bin, "python"),
        required = TRUE
    )
} else {
    tensorflow::use_python(
        file.path(conda_env_bin, "python"),
        required = TRUE
    )
}

# ---------------------------------------------------------
# BLOCK 4 — Define Directory Structure Under PIXI_WORKDIR
# ---------------------------------------------------------
# Create the standard PiXi directory tree under PIXI_WORKDIR:
#   Data/Training, Data/Testing
#   CPU/Training,  CPU/Testing
#   GPU/Training,  GPU/Testing
base <- Sys.getenv("PIXI_WORKDIR")
if (base == "") stop("PIXI_WORKDIR is not set.")
cat("PIXIRUN: base =", base, "\n")

dirs <- c(
    "Data/Training",
    "Data/Testing",
    "CPU/Training",
    "CPU/Testing",
    "GPU/Training",
    "GPU/Testing"
)

for (d in dirs) {
    full <- file.path(base, d)
    if (!dir.exists(full)) {
        dir.create(full, recursive = TRUE)
        cat("PIXIRUN: created dir:", full, "\n")
    }
}

# ---------------------------------------------------------
# BLOCK 5 — Select Mode-Specific Output Directories
# ---------------------------------------------------------
# Choose the output directories that correspond to the current mode.
# CPU mode → CPU/Training & CPU/Testing
# GPU mode → GPU/Training & GPU/Testing
if (pixi_mode == "gpu") {
    train_dir <- file.path(base, "GPU/Training")
    test_dir  <- file.path(base, "GPU/Testing")
} else {
    train_dir <- file.path(base, "CPU/Training")
    test_dir  <- file.path(base, "CPU/Testing")
}
cat("PIXIRUN: train_dir =", train_dir, "\n")
cat("PIXIRUN: test_dir  =", test_dir,  "\n")

# Load core PiXi function definitions
source(file.path(pixi_root, "GenerateTrainingData_Modular.R"))
source(file.path(pixi_root, "Classifier_Modular.R"))
source(file.path(pixi_root, "Predictor_Modular.R"))
source(file.path(pixi_root, "Testing_Classifier_Modular.R"))
source(file.path(pixi_root, "Testing_Predictor_Modular.R"))
source(file.path(pixi_root, "GenerateSequenceTrainingData_Modular.R"))

# ---------------------------------------------------------
# BLOCK 6 — Load Feature-Extractor and Ensemble Helpers
# ---------------------------------------------------------
# Source helper libraries required for feature-extractor extraction
# and ensemble training-data generation.
source(file.path(pixi_root, "feature_extractor_utils.R"))
source(file.path(pixi_root, "ensemble_training_data_generator.R"))

cat("\nPiXi environment activated in mode: ", pixi_mode, "\n")

# ---------------------------------------------------------
# BLOCK 7 — Clean Mode-Specific Output Directories
# ---------------------------------------------------------
# Remove any previous files so each run starts from a clean state.
cat("PIXIRUN: contents BEFORE CLEAN (train_dir):\n")
print(list.files(train_dir))
cat("PIXIRUN: contents BEFORE CLEAN (test_dir):\n")
print(list.files(test_dir))

unlink(list.files(train_dir, full.names = TRUE), recursive = TRUE)
unlink(list.files(test_dir,  full.names = TRUE), recursive = TRUE)

cat("PIXIRUN: contents AFTER CLEAN (train_dir):\n")
print(list.files(train_dir))
cat("PIXIRUN: contents AFTER CLEAN (test_dir):\n")
print(list.files(test_dir))

# ---------------------------------------------------------
# BLOCK 8 — Generate Training Data (Always in Data/ Directory)
# ---------------------------------------------------------
# Training data are always generated in base/Data/Training.
# They are later copied into the mode-specific CPU/ or GPU/ directories.
setwd(file.path(base, "Data/Training"))
cat("PIXIRUN: CWD for training data generation =", getwd(), "\n")

if (!file.exists(empirical_file)) {
    stop("Empirical features file not found:\n  ", empirical_file)
}

GenerateTrainingData_safe(
    m                     = m,
    Nk                    = Nk,
    training_prefix       = training_prefix,
    trait_type            = trait_type,
    logAlphaMin           = logAlphaMin,
    logAlphaMax           = logAlphaMax,
    logSigmaSqMin         = logSigmaSqMin,
    logSigmaSqMax         = logSigmaSqMax,
    empiricalFile         = empirical_file,
    zscore_features       = zscore_features,
    use_empirical_density = use_empirical_density,
    synthetic_pool_mode   = synthetic_pool_mode,
    IQR_K                 = IQR_K
)

cat("PIXIRUN: contents AFTER GenerateTrainingData_safe (Data/Training):\n")
print(list.files(file.path(base, "Data/Training")))

# ---------------------------------------------------------
# BLOCK 9 — Copy Generated Data into Mode-Specific Directories
# ---------------------------------------------------------
# Copy training files into the mode-specific Training directory and place
# the empirical Testing.features file into the mode-specific Testing directory.
# Guarantees that CPU and GPU runs operate on identical data.
train_src <- file.path(base, "Data/Training")
test_src  <- file.path(base, "Data/Testing")

train_files <- list.files(train_src, full.names = TRUE)

cat("PIXIRUN: train_src =", train_src, "\n")
cat("PIXIRUN: contents BEFORE COPY (train_src):\n")
print(list.files(train_src))
cat("PIXIRUN: contents BEFORE COPY (train_dir):\n")
print(list.files(train_dir))

file.copy(train_files, train_dir, overwrite = TRUE)

# Copy the single empirical Testing.features file
empirical_dir <- file.path(pixi_root, "Empirical")
empirical_features <- list.files(
    empirical_dir,
    pattern = "\\.features$",
    full.names = TRUE
)
if (length(empirical_features) != 1) {
    stop("Expected exactly one empirical .features file in:\n  ", empirical_dir)
}
if (!file.exists(empirical_features[1])) {
    stop("Empirical features file does not exist:\n  ", empirical_features[1])
}

file.copy(
    empirical_features,
    file.path(test_dir, paste0(testing_prefix, ".features")),
    overwrite = TRUE
)

cat("PIXIRUN: contents AFTER DATA COPY (train_dir):\n")
print(list.files(train_dir))
cat("PIXIRUN: contents AFTER DATA COPY (test_dir):\n")
print(list.files(test_dir))

# ---------------------------------------------------------
# BLOCK 10 — Run Classifier Training (Mode-Specific)
# ---------------------------------------------------------
# Train the neural-network classifier on the data now residing in train_dir.
setwd(train_dir)
cat("PIXIRUN: CWD BEFORE ClassifierCVnn_safe =", getwd(), "\n")
cat("PIXIRUN: contents BEFORE ClassifierCVnn_safe (train_dir):\n")
print(list.files(train_dir))

ClassifierCVnn_safe(
    num_layers      = num_layers_cls,
    batchsize       = batchsize,
    num_epochs      = num_epochs,
    log_lambda_min  = log_lambda_min,
    log_lambda_max  = log_lambda_max,
    num_lambda      = num_lambda,
    gamma_min       = gamma_min,
    gamma_max       = gamma_max,
    num_gamma       = num_gamma,
    training_prefix = training_prefix
)

# ---------------------------------------------------------
# BLOCK 11 — Extract and Save Frozen Feature Extractor
# ---------------------------------------------------------
# Freeze the trained classifier (minus final dense + activation layers)
# and save it as a feature extractor for later ensemble use.
cat("PIXIRUN: Extracting frozen feature extractor for base model...\n")

model_path <- file.path(train_dir,
                        paste0(training_prefix, ".nn.", num_layers_cls, ".classifier.keras"))
cat("Looking for base classifier model at:", model_path, "\n")

if (!file.exists(model_path)) {
    stop("ERROR: Base classifier model not found!\nExpected path: ", model_path,
         "\nMake sure ClassifierCVnn_safe completed successfully.")
}
save_feature_extractor(model_path = model_path)
cat("PIXIRUN: ✓ Feature extractor successfully saved.\n")

# ---------------------------------------------------------
# BLOCK 12 — Generate Training Classifications and Probabilities
# ---------------------------------------------------------
# Run the trained classifier on the training set itself to produce
# classification labels and class probabilities.
setwd(train_dir)
cat("PIXIRUN: Running TRAINING classification\n")
train_path <- file.path(train_dir, training_prefix)

PiXiClassifyNN_safe(
    training_prefix = train_path,
    testing_prefix  = train_path,   # classify the Training set
    num_layers      = num_layers_cls
)

cat("PIXIRUN: contents AFTER TRAINING classification (train_dir):\n")
print(list.files(train_dir))

# ---------------------------------------------------------
# BLOCK 13 — Run Predictor Training
# ---------------------------------------------------------
# Train the neural-network predictor (expression-optima regression).
PredictorCVnn_safe(
    num_layers      = num_layers_pred,
    batchsize       = batchsize,
    num_epochs      = num_epochs,
    log_lambda_min  = log_lambda_min,
    log_lambda_max  = log_lambda_max,
    num_lambda      = num_lambda,
    gamma_min       = gamma_min,
    gamma_max       = gamma_max,
    num_gamma       = num_gamma,
    training_prefix = training_prefix
)

# ---------------------------------------------------------
# BLOCK 14 — Run Testing Classification and Prediction
# ---------------------------------------------------------
# Apply the trained classifier and predictor to the empirical testing data.
# Full paths to the Training and Testing prefixes are required.
setwd(test_dir)
cat("PIXIRUN: CWD BEFORE PiXiClassifyNN_safe =", getwd(), "\n")
cat("PIXIRUN: contents BEFORE PiXiClassifyNN_safe (test_dir):\n")
print(list.files(test_dir))
cat("PIXIRUN: contents BEFORE PiXiClassifyNN_safe (train_dir):\n")
print(list.files(train_dir))

train_path <- file.path(train_dir, training_prefix)
test_path  <- file.path(test_dir,  testing_prefix)
cat("PIXIRUN: train_path =", train_path, "\n")
cat("PIXIRUN: test_path  =", test_path,  "\n")

# Testing classification
PiXiClassifyNN_safe(
    training_prefix = train_path,
    testing_prefix  = test_path,
    num_layers      = num_layers_cls
)

cat("PIXIRUN: contents AFTER PiXiClassifyNN_safe (test_dir):\n")
print(list.files(test_dir))
cat("PIXIRUN: contents AFTER PiXiClassifyNN_safe (train_dir):\n")
print(list.files(train_dir))

# Testing prediction
PiXiPredictNN(
    training_prefix = train_path,
    testing_prefix  = test_path,
    num_layers      = num_layers_pred
)

# ---------------------------------------------------------
# BLOCK 15 — Run Unified Plotting
# ---------------------------------------------------------
# Invoke the unified plotting orchestrator.
# Re-export environment variables so the plotting scripts can locate
# the correct run directory and mode.
cat("PIXIRUN: Running unified plotting...\n")

Sys.setenv(PIXI_WORKDIR = base)
Sys.setenv(PIXI_MODE    = pixi_mode)   # keep lower-case for consistency

system2(
    file.path(conda_env_bin, "Rscript"),
    args = c(file.path(pixi_root, "plot_all.R"),
             MODE, num_layers_cls, num_layers_pred)
)

cat("PIXIRUN: Plotting complete.\n")
cat("=== PiXi run complete ===\n")