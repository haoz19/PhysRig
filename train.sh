#!/bin/bash
# edit
# =============================================================================
# PhysRig Training/Validation Script for Sphere Dataset
# =============================================================================
#
# WORKFLOW EXPLANATION:
# This script validates cuboid parameters in cuboid_utils.py by:
#   1. Running INFERENCE to generate ground truth output based on skeleton velocities
#   2. Copying inference output to the gt/ folder expected by validation script
#   3. Running VALIDATION to compare simulation against inference output
#   4. If loss is LOW → cuboid parameters are CORRECT
#
# BEFORE RUNNING:
# - Modify parameters in exp_motion/train/cuboid_utils.py as needed
# - Parameters to tune are documented in log.md under "Default Cuboid Parameters"
#
# USAGE: bash train.sh
# Run from PhysRig directory
#
# =============================================================================

set -e  # Exit on error

# Start total timer
TOTAL_START_TIME=$(date +%s)

# Configuration
PYTHON=python
DATASET="dragon_tjx"
#SKELETON_DIR="data/dragon_tjx/skeleton_test/skeleton"
SKELETON_DIR="data/dragon_tjx/skeleton"
DATASET_DIR="data/${DATASET}"

# Cuboid update mode: "velocity_only", "location_only", or "both"
# - velocity_only: Update cuboid velocity from tracked points, keep location fixed
# - location_only: Update cuboid location from tracked points, keep velocity traditional
# - both: Update both velocity and location from tracked points
CUBOID_UPDATE_MODE="both"

# Position method for computing cuboid center from tracked points:
# - mean: Simple average of all tracked points (default, original behavior)
# - median: Median position of tracked points
# - weighted: Inverse distance weighted from initial cuboid center
# - bbox: Center of bounding box (min/max)
# - adaptive: Trimmed mean with outlier removal + density weighting
# - pca: Principal Component Analysis center (covariance-based)
# - optimized: LBFGS optimization to minimize L2 deviation (EXPENSIVE!)
POSITION_METHOD="adaptive"

# Check dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Error: Dataset directory '$DATASET_DIR' not found!"
    exit 1
fi

# =============================================================================
# STEP 1: Run Inference to Generate Ground Truth
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 1: Running Inference to Generate Ground Truth"
echo "============================================================"
echo ""

INFERENCE_SCRIPT="exp_motion/train/gt_code/baseline_gt_tjx.py"
INFERENCE_OUTPUT_DIR="output/inference/${DATASET}"
WANDB_NAME_INF="${DATASET}_inference"

# Inference parameters (adjust as needed)
NUM_FRAMES_INF=16           # Number of GT frames to process (must match skeleton files)
NUM_INTERMEDIATE_INF=8      # Intermediate frames (keep 0 to match training)
SUBSTEP_INF=100             # Simulation substeps per frame
YOUNGS_INF=6e4              # Young's modulus
NU_INF=0.3                  # Poisson's ratio
SAMPLE_PARTICLES_INF=100    # Number of sample particles
VELO_FACTOR_INF=1.0         # Velocity factor (1.0 = use GT velocities)

echo "Running inference with parameters:"
echo "  - Dataset: $DATASET_DIR"
echo "  - Skeleton: $SKELETON_DIR"
echo "  - Output: $INFERENCE_OUTPUT_DIR"
echo "  - Frames: $NUM_FRAMES_INF"
echo "  - Cuboid Update Mode: $CUBOID_UPDATE_MODE"
echo "  - Position Method: $POSITION_METHOD"
echo ""

# Clean old inference outputs to avoid confusion
echo "Cleaning old inference outputs..."
rm -rf "$INFERENCE_OUTPUT_DIR"/*

INFERENCE_START_TIME=$(date +%s)

$PYTHON $INFERENCE_SCRIPT \
    --dataset_dir $DATASET_DIR \
    --skeleton_dir $SKELETON_DIR \
    --wandb_name $WANDB_NAME_INF \
    --num_frames $NUM_FRAMES_INF \
    --num_intermediate_frames $NUM_INTERMEDIATE_INF \
    --substep $SUBSTEP_INF \
    --youngs $YOUNGS_INF \
    --nu $NU_INF \
    --sample_particles $SAMPLE_PARTICLES_INF \
    --velo_factor $VELO_FACTOR_INF \
    --output_dir $INFERENCE_OUTPUT_DIR \
    --cuboid_update_mode $CUBOID_UPDATE_MODE \
    --position_method $POSITION_METHOD

INFERENCE_END_TIME=$(date +%s)
INFERENCE_DURATION=$((INFERENCE_END_TIME - INFERENCE_START_TIME))

echo ""
echo "✅ Inference complete! (Time: ${INFERENCE_DURATION}s)"
echo ""

# =============================================================================
# STEP 2: Copy Inference Output to GT Directory
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 2: Setting Up Ground Truth from Inference Output"
echo "============================================================"
echo ""

# Find the newly created inference output subdirectory
INFERENCE_SUBDIR=$(ls -td "$INFERENCE_OUTPUT_DIR"/*/ 2>/dev/null | head -1)

if [ -z "$INFERENCE_SUBDIR" ]; then
    echo "❌ Error: No inference output found in '$INFERENCE_OUTPUT_DIR'"
    exit 1
fi

# The inference outputs PLY files to an 'output' subdirectory within the run folder
INFERENCE_PLY_DIR="${INFERENCE_SUBDIR}output"

if [ ! -d "$INFERENCE_PLY_DIR" ]; then
    # Try alternative: direct output folder
    INFERENCE_PLY_DIR="$INFERENCE_SUBDIR"
fi

echo "Inference PLY directory: $INFERENCE_PLY_DIR"

# Create gt/ directory in dataset (clear old files first!)
GT_DIR="${DATASET_DIR}/gt"
rm -rf "$GT_DIR"
mkdir -p "$GT_DIR"

# Copy inference output PLY files to gt/ directory
echo "Copying inference output to $GT_DIR..."
if ls "$INFERENCE_PLY_DIR"/gt_*.ply 1>/dev/null 2>&1; then
    cp "$INFERENCE_PLY_DIR"/gt_*.ply "$GT_DIR/"
    echo "✅ Copied gt_*.ply files to $GT_DIR"
else
    echo "⚠️  No gt_*.ply files found in inference output."
    echo "   Checking for existing GT files in dataset..."
    
    # Check if sphere/output already has GT files we can use
    if ls "${DATASET_DIR}/output"/gt_*.ply 1>/dev/null 2>&1; then
        echo "   Using existing GT files from ${DATASET_DIR}/output/"
        cp "${DATASET_DIR}/output"/gt_*.ply "$GT_DIR/"
    else
        echo "❌ Error: No GT files available!"
        exit 1
    fi
fi

# Count GT frames
NUM_GT_FILES=$(ls -1 "$GT_DIR"/gt_*.ply 2>/dev/null | wc -l)
echo "Total GT frames available: $NUM_GT_FILES"
echo ""

# =============================================================================
# STEP 3: Run Validation (Training)
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 3: Running Validation Against Inference Output"
echo "============================================================"
echo ""

TRAIN_SCRIPT="exp_motion/train/spv_code/baseline_spv_bm_pu_tjx_SGD.py"
TRAIN_OUTPUT_DIR="output/train/${DATASET}"
WANDB_NAME_TRAIN="${DATASET}_validation"

# Validation parameters (should match inference where applicable)
NUM_FRAMES_TRAIN=$NUM_GT_FILES  # Use all available GT frames
SUBSTEP_TRAIN=100               # Same as inference
YOUNGS_TRAIN=6e4                # Same as inference
NU_TRAIN=0.3                    # Same as inference
SAMPLE_PARTICLES_TRAIN=100      # Same as inference
VELO_FACTOR_TRAIN=0.0           # Start from zero velocity (will learn)

# Training-specific parameters
TRAIN_ITERS=100                  # Total iterations (50 is enough for validation; min loss usually found early)
ITER_MATERIAL=10                # Material training iteration threshold
LR=0.01                         # Learning rate
MAX_GRAD_NORM=1.0               # Gradient clipping
WARMUP_STEP=5                   # Warmup steps
STRIDE=1                        # Temporal stride

echo "Running validation with parameters:"
echo "  - Dataset: $DATASET_DIR"
echo "  - GT Directory: $GT_DIR"
echo "  - Skeleton init: $SKELETON_DIR"
echo "  - Output: $TRAIN_OUTPUT_DIR"
echo "  - Frames: $NUM_FRAMES_TRAIN"
echo "  - Iterations: $TRAIN_ITERS"
echo ""

VALIDATION_START_TIME=$(date +%s)

$PYTHON $TRAIN_SCRIPT \
    --dataset_dir $DATASET_DIR \
    --skeleton_init_dir $SKELETON_DIR \
    --wandb_name $WANDB_NAME_TRAIN \
    --num_frames $NUM_FRAMES_TRAIN \
    --substep $SUBSTEP_TRAIN \
    --youngs $YOUNGS_TRAIN \
    --nu $NU_TRAIN \
    --sample_particles $SAMPLE_PARTICLES_TRAIN \
    --velo_factor $VELO_FACTOR_TRAIN \
    --output_dir $TRAIN_OUTPUT_DIR \
    --train_iters $TRAIN_ITERS \
    --iter_material $ITER_MATERIAL \
    --lr $LR \
    --max_grad_norm $MAX_GRAD_NORM \
    --warmup_step $WARMUP_STEP \
    --stride $STRIDE \
    --cuboid_update_mode $CUBOID_UPDATE_MODE \
    --position_method $POSITION_METHOD \
    $CAPSULE_ARGS

VALIDATION_END_TIME=$(date +%s)
VALIDATION_DURATION=$((VALIDATION_END_TIME - VALIDATION_START_TIME))

# Calculate total time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "============================================================"
echo "VALIDATION COMPLETE!"
echo "============================================================"
echo ""
echo "TIMING:"
echo "  - Inference:  ${INFERENCE_DURATION}s"
echo "  - Validation: ${VALIDATION_DURATION}s"
echo "  - TOTAL:      ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "Results saved to: $TRAIN_OUTPUT_DIR"
echo ""

# ============================================================
# STEP 4: Generate Cuboid Sphere Meshes
# ============================================================

echo "============================================================"
echo "Generating Cuboid Sphere Meshes"
echo "============================================================"
echo ""

# Find the most recent training output directory
LATEST_TRAIN_DIR=$(ls -td "${TRAIN_OUTPUT_DIR}"/*/ 2>/dev/null | head -1)

if [ -d "$LATEST_TRAIN_DIR" ]; then
    SKELETON_OUTPUT_DIR="${LATEST_TRAIN_DIR}skeleton"
    CUBOID_SIZES_FILE="${LATEST_TRAIN_DIR}cuboid_sizes.npy"
    CUBOID_OUTPUT_DIR="${LATEST_TRAIN_DIR}cuboid"
    
    if [ -f "$CUBOID_SIZES_FILE" ] && [ -d "$SKELETON_OUTPUT_DIR" ]; then
        echo "Skeleton directory: $SKELETON_OUTPUT_DIR"
        echo "Cuboid sizes file: $CUBOID_SIZES_FILE"
        echo "Output directory: $CUBOID_OUTPUT_DIR"
        echo ""
        
        $PYTHON exp_motion/utils/generate_cuboid_meshes.py \
            --skeleton_dir "$SKELETON_OUTPUT_DIR" \
            --cuboid_sizes_file "$CUBOID_SIZES_FILE" \
            --output_dir "$CUBOID_OUTPUT_DIR" \
            --subdivisions 2
        
        echo ""
        echo "✅ Cuboid meshes generated in: $CUBOID_OUTPUT_DIR"
    else
        echo "⚠️  Warning: Could not find cuboid_sizes.npy or skeleton directory"
        echo "   Skipping cuboid mesh generation"
    fi
else
    echo "⚠️  Warning: Could not find training output directory"
    echo "   Skipping cuboid mesh generation"
fi

echo ""

# ============================================================
# Loss Statistics Summary
# ============================================================

# Find the most recent frame_losses.txt in subdirectories
FRAME_LOSS_FILE=$(ls -t "${TRAIN_OUTPUT_DIR}"/*/frame_losses.txt 2>/dev/null | head -1)

if [ -f "$FRAME_LOSS_FILE" ]; then
    echo "============================================================"
    echo "Training Loss Statistics"
    echo "============================================================"
    
    # Parse loss statistics and output as table
    awk 'NR>1 {
        losses[NR-1] = $2
        sum += $2
        if (NR==2) {
            initial = $2
            min = $2
            min_iter = 0
        }
        if ($2 < min) {
            min = $2
            min_iter = NR-2
        }
        count = NR-1
    }
    END {
        final = losses[count]
        avg = sum / count
        printf "| best loss (iter) | avg loss | final   | initial |\n"
        printf "| ---------------- | -------- | ------- | ------- |\n"
        printf "| %.2f (iter %d)   | %.2f     | %.2f    | %.2f    |\n", min, min_iter, avg, final, initial
    }' "$FRAME_LOSS_FILE"
    
    echo "============================================================"
    echo ""
else
    echo "Warning: No frame_losses.txt found in ${TRAIN_OUTPUT_DIR}/*/"
    echo ""
fi

echo "INTERPRETATION:"
echo "  - Check 'loss_array.npy' and 'frame_loss_plot.png' for loss values"
echo "  - LOW LOSS = cuboid parameters in cuboid_utils.py are CORRECT"
echo "  - HIGH LOSS = cuboid parameters need adjustment"
echo ""
echo "To modify cuboid parameters, edit: exp_motion/train/cuboid_utils.py"
echo "See log.md for parameter documentation and tuning history."
echo ""
