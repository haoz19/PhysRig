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
DATASET="mixamo_walk"
SKELETON_DIR="data/${DATASET}/skeleton"
DATASET_DIR="data/${DATASET}"

# Cuboid update mode: "none", "velocity_only", "location_only", or "both"
# - none: Traditional velocity (closest-point), fixed location (no tracked-point updates)
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

# Cuboid sizing strategy: "fixed", "adaptive", "knn", "hybrid", "ceil_and_floor", "raycast"
# - fixed: Uniform radius from global min inter-vertex distance (original behavior)
# - adaptive: Per-cuboid radius from distance to nearest neighbor cuboid
# - knn: Radius = distance to K-th nearest dense point (coeff HARDCODED to 1.0)
# - hybrid: KNN (coeff=1.0) capped by (global_min_ctrl_dist / 2) * coeff
# - ceil_and_floor: KNN clamped between grid_dx*sqrt(5) and (global_min_ctrl_dist/2)*coeff
# - raycast: Directional sphere expansion with density-based boundary detection
CUBOID_SIZE_MODE="fixed"

# Scaling coefficient for cuboid radius:
# - knn mode: IGNORED (hardcoded to 1.0)
# - hybrid mode: cap = (global_min_ctrl_dist / 2) * coeff
# - ceil_and_floor mode: ceiling = (global_min_ctrl_dist / 2) * coeff; floor = grid_dx * sqrt(5)
# - other modes: scales the radius as before
CUBOID_SIZE_COEFF=0.1

# K for KNN-based sizing modes (knn, hybrid, raycast)
CUBOID_KNN_K=56

# Dynamic cuboid radius capping (velocity-divergence based overlap prevention)
DYNAMIC_CUBOID_CAP=true

# Clamp cuboid radii to at least grid_dx
CLAMP_CUBOID_MIN_RADIUS=true

# Force regeneration: delete and regenerate skeleton and infilled data every run
# Set to true to always start fresh (useful when upstream data or parameters change)
FORCE_REGENERATE=false

# Script paths
INFERENCE_SCRIPT="exp_motion/train/gt_code/baseline_gt_tjx.py"
INFERENCE_OUTPUT_DIR="output/inference/${DATASET}"
WANDB_NAME_INF="${DATASET}_inference"

# Inference parameters (adjust as needed)
NUM_INTERMEDIATE_INF=0      # Intermediate frames (keep 0 to match training)
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

# Wipe skeleton and infilled data when FORCE_REGENERATE is enabled
SKELETON_MESH_DIR="${DATASET_DIR}/skeleton_mesh"
INFILLED_DIR="${DATASET_DIR}/infilled"

if [ "$FORCE_REGENERATE" = true ]; then
    echo ""
    echo "FORCE_REGENERATE=true — deleting existing skeleton and infilled data"
    rm -rf "$SKELETON_DIR"
    rm -rf "$INFILLED_DIR"
    rm -rf "${DATASET_DIR}/gt"
fi

# Generate skeleton PLY files if missing
if [ ! -d "$SKELETON_DIR" ] || [ $(ls -1 "$SKELETON_DIR"/gt_*.ply 2>/dev/null | wc -l) -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Generating skeleton PLY files from skeleton_mesh"
    echo "============================================================"
    echo ""
    
    # Check if skeleton_mesh directory exists
    if [ ! -d "$SKELETON_MESH_DIR" ]; then
        echo "❌ Error: skeleton_mesh directory not found: $SKELETON_MESH_DIR"
        echo "   Run fbx_to_pipeline.py or sim.sh (full pipeline) first."
        exit 1
    fi
    
    # Count skeleton mesh OBJ files
    SKEL_MESH_COUNT=$(ls -1 "$SKELETON_MESH_DIR"/skeleton_frame_*.obj 2>/dev/null | wc -l)
    if [ "$SKEL_MESH_COUNT" -eq 0 ]; then
        echo "❌ Error: No skeleton_frame_*.obj files found in $SKELETON_MESH_DIR"
        echo "   Run fbx_to_pipeline.py or sim.sh (full pipeline) first."
        exit 1
    fi
    
    echo "Found $SKEL_MESH_COUNT skeleton mesh OBJ files"
    echo "Generating skeleton centroids..."
    
    mkdir -p "$SKELETON_DIR"
    rm -f "$SKELETON_DIR"/gt_*.ply
    
    SKEL_START_TIME=$(date +%s)
    
    $PYTHON exp_motion/utils/generate_skeleton_centroids.py \
        --input_folder "$DATASET_DIR" \
        --output_folder "$SKELETON_DIR" \
        --num_frames "$SKEL_MESH_COUNT"
    
    SKEL_END_TIME=$(date +%s)
    SKEL_DURATION=$((SKEL_END_TIME - SKEL_START_TIME))
    
    # Verify skeleton PLY files were created
    SKEL_PLY_COUNT=$(ls -1 "$SKELETON_DIR"/gt_*.ply 2>/dev/null | wc -l)
    if [ "$SKEL_PLY_COUNT" -eq 0 ]; then
        echo "❌ Error: generate_skeleton_centroids.py failed to create skeleton PLY files"
        exit 1
    fi
    
    echo ""
    echo "✅ Skeleton generation complete! (Time: ${SKEL_DURATION}s)"
    echo "  Generated $SKEL_PLY_COUNT skeleton PLY files in $SKELETON_DIR"
    echo ""
else
    echo "✓ Skeleton directory found: $SKELETON_DIR"
fi

# Auto-detect number of frames from skeleton files
NUM_FRAMES_INF=$(ls -1 "$SKELETON_DIR"/gt_*.ply 2>/dev/null | wc -l)
if [ "$NUM_FRAMES_INF" -eq 0 ]; then
    echo "❌ Error: No skeleton files (gt_*.ply) found in $SKELETON_DIR"
    echo "   This should not happen - skeleton generation failed silently."
    exit 1
fi

# Cap total simulated frames (num_frames + intermediate - 1) at 80
TOTAL_SIM_FRAMES=$((NUM_FRAMES_INF + NUM_INTERMEDIATE_INF - 1))
if [ "$TOTAL_SIM_FRAMES" -gt 80 ]; then
    NUM_FRAMES_INF=$((80 - NUM_INTERMEDIATE_INF + 1))
    echo "Capping: total sim frames would be $TOTAL_SIM_FRAMES, reducing num_frames to $NUM_FRAMES_INF (total = 80)"
fi

# Generate infilled data if missing
if [ ! -d "$INFILLED_DIR" ]; then
    echo ""
    echo "============================================================"
    echo "Generating infilled data via infill.sh"
    echo "============================================================"
    echo ""
    
    INFILL_START_TIME=$(date +%s)
    
    bash infill.sh \
        --input_folder "$DATASET_DIR" \
        --output_folder_infilled "$INFILLED_DIR" \
        --output_folder_gt "${DATASET_DIR}/gt" \
        --resolution 20 \
        --samples_per_voxel 5
    
    INFILL_END_TIME=$(date +%s)
    INFILL_DURATION=$((INFILL_END_TIME - INFILL_START_TIME))
    
    # Verify infilled directory was created
    if [ ! -d "$INFILLED_DIR" ]; then
        echo "❌ Error: infill.sh failed to create infilled directory"
        exit 1
    fi
    
    echo ""
    echo "✅ Infill complete! (Time: ${INFILL_DURATION}s)"
    echo ""
else
    echo "✓ Infilled directory found: $INFILLED_DIR"
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

# Build optional flags
EXTRA_ARGS=""
[ "$DYNAMIC_CUBOID_CAP" = true ] && EXTRA_ARGS="$EXTRA_ARGS --dynamic_cuboid_cap"
[ "$CLAMP_CUBOID_MIN_RADIUS" = true ] && EXTRA_ARGS="$EXTRA_ARGS --clamp_cuboid_min_radius"

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
    --cuboid_knn_k $CUBOID_KNN_K \
    $EXTRA_ARGS

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

# =============================================================================
# Inference Stats Table
# =============================================================================
MESH_VERTICES=$(grep -c '^v ' "$MESH_PATH" 2>/dev/null || echo 0)
STATS_FILE=$(ls -t "$INFERENCE_OUTPUT_DIR"/*/inference_stats.json 2>/dev/null | head -1)
if [ -f "$STATS_FILE" ]; then
    $PYTHON -c "
import json, sys
mesh_verts = int(sys.argv[2])
with open(sys.argv[1]) as f:
    s = json.load(f)
h = '| {:>8} | {:>7} | {:>6} | {:>6} | {:>14} | {:>5} |'
print(h.format('Vertices', 'Cuboids', 'Points', 'Frames', 'Inference Time', 'FPS'))
print('| -------- | ------- | ------ | ------ | -------------- | ----- |')
print(h.format(mesh_verts, s['cuboids'], s['points'], s['frames'],
               str(s['inference_time_s']) + 's', s['fps']))
" "$STATS_FILE" "$MESH_VERTICES"
    echo ""
fi
