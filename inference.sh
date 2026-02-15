#!/bin/bash

# =============================================================================
# PhysRig Inference Script for Sphere Dataset
# =============================================================================
#
# This script runs inference to generate ground truth output.
# The output is based on skeleton velocities and material simulation.
#
# USAGE: bash inference.sh
# Run from PhysRig directory
#
# =============================================================================

set -e  # Exit on error

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

# Cuboid sizing strategy: "fixed", "adaptive", "knn", "hybrid", "raycast"
# - fixed: Uniform radius from global min inter-vertex distance (original behavior)
# - adaptive: Per-cuboid radius from distance to nearest neighbor cuboid
# - knn: Radius = distance to K-th nearest dense point (coeff HARDCODED to 1.0)
# - hybrid: KNN (coeff=1.0) capped by fixed radius (= coeff * grid_dx)
# - raycast: Directional sphere expansion with density-based boundary detection
CUBOID_SIZE_MODE="fixed"

# Scaling coefficient for cuboid radius:
# - knn mode: IGNORED (hardcoded to 1.0)
# - hybrid mode: controls the fixed radius constraint (fixed_r = coeff * grid_dx)
# - other modes: scales the radius as before
CUBOID_SIZE_COEFF=0.8

# K for KNN-based sizing modes (knn, hybrid, raycast)
CUBOID_KNN_K=56

# Script paths
INFERENCE_SCRIPT="exp_motion/train/gt_code/baseline_gt_tjx.py"
INFERENCE_OUTPUT_DIR="output/inference/${DATASET}"
WANDB_NAME_INF="${DATASET}_inference"

# Inference parameters (adjust as needed)
NUM_FRAMES_INF=80           # Number of GT frames to process (must match skeleton files)
NUM_INTERMEDIATE_INF=8      # Intermediate frames (keep 0 to match training)
SUBSTEP_INF=100             # Simulation substeps per frame
YOUNGS_INF=6e4              # Young's modulus
NU_INF=0.3                  # Poisson's ratio
SAMPLE_PARTICLES_INF=100    # Number of sample particles
VELO_FACTOR_INF=1.0         # Velocity factor (1.0 = use GT velocities)

# Check dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Error: Dataset directory '$DATASET_DIR' not found!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Running Inference for: $DATASET"
echo "============================================================"
echo ""
echo "Running inference with parameters:"
echo "  - Dataset: $DATASET_DIR"
echo "  - Skeleton: $SKELETON_DIR"
echo "  - Output: $INFERENCE_OUTPUT_DIR"
echo "  - Frames: $NUM_FRAMES_INF"
echo "  - Cuboid Update Mode: $CUBOID_UPDATE_MODE"
echo "  - Position Method: $POSITION_METHOD"
echo "  - Cuboid Size Mode: $CUBOID_SIZE_MODE"
echo "  - Cuboid Size Coeff: $CUBOID_SIZE_COEFF"
echo "  - Cuboid KNN K: $CUBOID_KNN_K"
echo ""

# Clean old inference outputs to avoid confusion
echo "Cleaning old inference outputs..."
rm -rf "$INFERENCE_OUTPUT_DIR"/*

INFERENCE_START_TIME=$(date +%s)

# Run inference
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
    --position_method $POSITION_METHOD \
    --cuboid_size_mode $CUBOID_SIZE_MODE \
    --cuboid_size_coeff $CUBOID_SIZE_COEFF \
    --cuboid_knn_k $CUBOID_KNN_K

INFERENCE_END_TIME=$(date +%s)
INFERENCE_DURATION=$((INFERENCE_END_TIME - INFERENCE_START_TIME))

echo ""
echo "✅ Inference complete! (Time: ${INFERENCE_DURATION}s)"
echo ""

# =============================================================================
# STEP 2: Locate Inference PLY Output
# =============================================================================

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

# Verify PLY files exist
PLY_COUNT=$(ls -1 "$INFERENCE_PLY_DIR"/gt_*.ply 2>/dev/null | wc -l)
if [ "$PLY_COUNT" -eq 0 ]; then
    echo "❌ Error: No gt_*.ply files found in $INFERENCE_PLY_DIR"
    exit 1
fi
echo "Found $PLY_COUNT inference PLY frames"
echo ""

# =============================================================================
# STEP 3: Apply Deformation (PLY -> Mesh Animation)
# =============================================================================
echo ""
echo "============================================================"
echo "Applying Deformation to Mesh"
echo "============================================================"
echo ""

DEFORM_SCRIPT="exp_motion/utils/apply_deformation.py"
MESH_PATH="${DATASET_DIR}/mesh/mesh_frame_0000.obj"
ANIMATION_OUTPUT_DIR="output/animation/${DATASET}"

if [ ! -f "$MESH_PATH" ]; then
    echo "❌ Error: Reference mesh not found: $MESH_PATH"
    exit 1
fi

echo "Reference mesh: $MESH_PATH"
echo "PLY source:     $INFERENCE_PLY_DIR"
echo "Output:         $ANIMATION_OUTPUT_DIR"
echo ""

# Clean old animation outputs
rm -rf "$ANIMATION_OUTPUT_DIR"
mkdir -p "$ANIMATION_OUTPUT_DIR"

DEFORM_START_TIME=$(date +%s)

$PYTHON $DEFORM_SCRIPT \
    --mesh_path "$MESH_PATH" \
    --ply_dir "$INFERENCE_PLY_DIR" \
    --output_dir "$ANIMATION_OUTPUT_DIR"

DEFORM_END_TIME=$(date +%s)
DEFORM_DURATION=$((DEFORM_END_TIME - DEFORM_START_TIME))

# Count output frames
ANIM_COUNT=$(ls -1 "$ANIMATION_OUTPUT_DIR"/mesh_frame_*.obj 2>/dev/null | wc -l)

echo ""
echo "✅ Deformation complete! (Time: ${DEFORM_DURATION}s)"
echo "  Generated $ANIM_COUNT deformed mesh frames"
echo "  Output: $ANIMATION_OUTPUT_DIR"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "TIMING:"
echo "  - Inference:    ${INFERENCE_DURATION}s"
echo "  - Deformation:  ${DEFORM_DURATION}s"
echo ""
echo "OUTPUTS:"
echo "  - Inference PLY:   $INFERENCE_PLY_DIR"
echo "  - Animation OBJ:   $ANIMATION_OUTPUT_DIR"
echo ""
