#!/bin/bash
# =============================================================================
# PhysRig Simulation Pipeline (FBX Input)
# =============================================================================
#
# End-to-end pipeline for FBX-based datasets:
#   1. FBX Conversion  - Extract mesh/ and skeleton_mesh/ OBJ sequences from FBX
#   2. Skeleton Points  - Generate skeleton centroid PLYs from skeleton_mesh/
#   3. Mesh Infill      - Generate infilled/ and gt/ point clouds from mesh/
#   4. Inference        - Run physics inference + validation + cuboid generation
#
# USAGE:
#   bash sim.sh --dataset <name> --fbx <path> [OPTIONS]
#
# EXAMPLES:
#   # Full pipeline from FBX:
#   bash sim.sh --dataset wyvern --fbx data/wyvern/Wyvern-Fly.fbx
#
#   # Inference only (data already prepared):
#   bash sim.sh --dataset wyvern
#
#   # Custom object names in FBX:
#   bash sim.sh --dataset wyvern --fbx data/wyvern/Wyvern-Fly.fbx \
#       --mesh_obj Body --skeleton_obj Skeleton
#
# Run from PhysRig directory
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Default Parameters
# =============================================================================

# Required
DATASET=""
FBX_PATH=""

# FBX object names
MESH_OBJ="U3DMesh"
SKELETON_OBJ="Armature"

# Pipeline control
PYTHON=python

# Infill parameters
RESOLUTION=20
SAMPLES_PER_VOXEL=5

# Cuboid parameters (same defaults as train.sh)
CUBOID_UPDATE_MODE="both"
POSITION_METHOD="adaptive"
CUBOID_SIZE_MODE="hybrid"
CUBOID_SIZE_COEFF=0.9
CUBOID_KNN_K=80

# Inference parameters
NUM_FRAMES_INF=16
NUM_INTERMEDIATE_INF=8
SUBSTEP_INF=100
YOUNGS_INF=6e4
NU_INF=0.3
SAMPLE_PARTICLES_INF=100
VELO_FACTOR_INF=1.0

# Validation parameters
TRAIN_ITERS=100
ITER_MATERIAL=10
LR=0.01
MAX_GRAD_NORM=1.0
WARMUP_STEP=5
STRIDE=1

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --fbx)
            FBX_PATH="$2"
            shift 2
            ;;
        --mesh_obj)
            MESH_OBJ="$2"
            shift 2
            ;;
        --skeleton_obj)
            SKELETON_OBJ="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --samples_per_voxel)
            SAMPLES_PER_VOXEL="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES_INF="$2"
            shift 2
            ;;
        --substep)
            SUBSTEP_INF="$2"
            shift 2
            ;;
        --youngs)
            YOUNGS_INF="$2"
            shift 2
            ;;
        --nu)
            NU_INF="$2"
            shift 2
            ;;
        --sample_particles)
            SAMPLE_PARTICLES_INF="$2"
            shift 2
            ;;
        --cuboid_update_mode)
            CUBOID_UPDATE_MODE="$2"
            shift 2
            ;;
        --position_method)
            POSITION_METHOD="$2"
            shift 2
            ;;
        --cuboid_size_mode)
            CUBOID_SIZE_MODE="$2"
            shift 2
            ;;
        --cuboid_size_coeff)
            CUBOID_SIZE_COEFF="$2"
            shift 2
            ;;
        --cuboid_knn_k)
            CUBOID_KNN_K="$2"
            shift 2
            ;;
        --train_iters)
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash sim.sh --dataset <name> --fbx <path> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --dataset NAME                Dataset name (e.g., wyvern)"
            echo "  --fbx PATH                    Path to FBX file (required for full pipeline)"
            echo ""
            echo "FBX Options:"
            echo "  --mesh_obj NAME               Mesh object name in FBX (default: U3DMesh)"
            echo "  --skeleton_obj NAME            Armature object name in FBX (default: Armature)"
            echo ""
            echo "Infill Options:"
            echo "  --resolution N                 Voxel grid resolution (default: 20)"
            echo "  --samples_per_voxel N          Samples per voxel (default: 5)"
            echo ""
            echo "Inference/Training Options:"
            echo "  --num_frames N                 Number of frames for inference (default: 16)"
            echo "  --substep N                    Simulation substeps (default: 100)"
            echo "  --youngs VAL                   Young's modulus (default: 6e4)"
            echo "  --nu VAL                       Poisson's ratio (default: 0.3)"
            echo "  --sample_particles N           Sample particles (default: 100)"
            echo "  --cuboid_update_mode MODE      velocity_only|location_only|both (default: both)"
            echo "  --position_method METHOD       mean|median|weighted|bbox|adaptive|pca (default: adaptive)"
            echo "  --cuboid_size_mode MODE        fixed|adaptive|knn|hybrid|raycast (default: hybrid)"
            echo "  --cuboid_size_coeff VAL        Cuboid radius coefficient (default: 0.9)"
            echo "  --cuboid_knn_k N               KNN K for cuboid sizing (default: 80)"
            echo "  --train_iters N                Training iterations (default: 100)"
            echo ""
            echo "Examples:"
            echo "  bash sim.sh --dataset wyvern --fbx data/wyvern/Wyvern-Fly.fbx"
            echo "  bash sim.sh --dataset wyvern   # inference only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Arguments
# =============================================================================

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required"
    echo "Use --help for usage information"
    exit 1
fi

DATASET_DIR="data/${DATASET}"
SKELETON_DIR="${DATASET_DIR}/skeleton"

# Get the script directory (PhysRig root) and add it to PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# =============================================================================
# Mode Selection (Interactive Prompt)
# =============================================================================

echo ""
echo "============================================================"
echo "  PhysRig Simulation Pipeline"
echo "  Dataset: ${DATASET}"
echo "============================================================"
echo ""
echo "Select mode:"
echo "  1) Full pipeline (FBX conversion + skeleton + infill + inference)"
echo "  2) Inference only (assumes data already prepared)"
echo ""
read -p "Enter choice [1/2]: " MODE_CHOICE

case $MODE_CHOICE in
    1)
        RUN_MODE="full"
        echo ""
        echo ">> Running FULL PIPELINE"
        ;;
    2)
        RUN_MODE="inference"
        echo ""
        echo ">> Running INFERENCE ONLY"
        ;;
    *)
        echo "Invalid choice: $MODE_CHOICE"
        exit 1
        ;;
esac

# Start total timer
TOTAL_START_TIME=$(date +%s)

# =============================================================================
# STEP 1: FBX Conversion (Full Pipeline Only)
# =============================================================================

if [ "$RUN_MODE" = "full" ]; then
    echo ""
    echo "============================================================"
    echo "STEP 1: FBX Conversion"
    echo "============================================================"
    echo ""

    # Validate FBX path
    if [ -z "$FBX_PATH" ]; then
        echo "ERROR: --fbx is required for full pipeline mode"
        exit 1
    fi

    if [ ! -f "$FBX_PATH" ]; then
        echo "ERROR: FBX file not found: $FBX_PATH"
        exit 1
    fi

    # Check that blender is available
    if ! command -v blender &> /dev/null; then
        echo "ERROR: 'blender' not found in PATH"
        echo "Please install Blender or add it to your PATH"
        exit 1
    fi

    echo "FBX file:        $FBX_PATH"
    echo "Output dir:      $DATASET_DIR"
    echo "Mesh object:     $MESH_OBJ"
    echo "Skeleton object: $SKELETON_OBJ"
    echo ""

    # Create dataset directory
    mkdir -p "$DATASET_DIR"

    FBX_START_TIME=$(date +%s)

    blender --background --python motionrep/datatools/fbx_to_pipeline.py -- \
        "$FBX_PATH" "$DATASET_DIR" \
        --mesh_obj "$MESH_OBJ" \
        --skeleton_obj "$SKELETON_OBJ"

    FBX_END_TIME=$(date +%s)
    FBX_DURATION=$((FBX_END_TIME - FBX_START_TIME))

    # Validate output
    MESH_COUNT=$(ls "$DATASET_DIR/mesh"/mesh_frame_*.obj 2>/dev/null | wc -l)
    SKEL_COUNT=$(ls "$DATASET_DIR/skeleton_mesh"/skeleton_frame_*.obj 2>/dev/null | wc -l)

    if [ "$MESH_COUNT" -eq 0 ]; then
        echo "ERROR: No mesh OBJs generated in $DATASET_DIR/mesh/"
        exit 1
    fi
    if [ "$SKEL_COUNT" -eq 0 ]; then
        echo "ERROR: No skeleton mesh OBJs generated in $DATASET_DIR/skeleton_mesh/"
        exit 1
    fi

    echo ""
    echo "FBX conversion complete! (Time: ${FBX_DURATION}s)"
    echo "  Mesh frames:     $MESH_COUNT"
    echo "  Skeleton frames: $SKEL_COUNT"
    echo ""

    # =========================================================================
    # STEP 2: Skeleton Points Generation
    # =========================================================================
    echo ""
    echo "============================================================"
    echo "STEP 2: Skeleton Points Generation"
    echo "============================================================"
    echo ""

    # Auto-detect number of frames from skeleton_mesh
    NUM_SKEL_FRAMES=$SKEL_COUNT

    echo "Input:       $DATASET_DIR/skeleton_mesh/"
    echo "Output:      $SKELETON_DIR/"
    echo "Num frames:  $NUM_SKEL_FRAMES"
    echo ""

    mkdir -p "$SKELETON_DIR"
    rm -f "$SKELETON_DIR"/gt_*.ply

    SKEL_START_TIME=$(date +%s)

    $PYTHON exp_motion/utils/generate_skeleton_centroids.py \
        --input_folder "$DATASET_DIR" \
        --output_folder "$SKELETON_DIR" \
        --num_frames "$NUM_SKEL_FRAMES"

    SKEL_END_TIME=$(date +%s)
    SKEL_DURATION=$((SKEL_END_TIME - SKEL_START_TIME))

    # Validate output
    SKEL_PLY_COUNT=$(ls "$SKELETON_DIR"/gt_*.ply 2>/dev/null | wc -l)
    if [ "$SKEL_PLY_COUNT" -eq 0 ]; then
        echo "ERROR: No skeleton PLYs generated in $SKELETON_DIR/"
        exit 1
    fi

    echo ""
    echo "Skeleton points complete! (Time: ${SKEL_DURATION}s)"
    echo "  Generated: $SKEL_PLY_COUNT centroid PLYs in $SKELETON_DIR/"
    echo ""

    # =========================================================================
    # STEP 3: Mesh Infilling
    # =========================================================================
    echo ""
    echo "============================================================"
    echo "STEP 3: Mesh Infilling"
    echo "============================================================"
    echo ""

    OUTPUT_FOLDER_INFILLED="${DATASET_DIR}/infilled"
    OUTPUT_FOLDER_GT="${DATASET_DIR}/gt"

    echo "Input:           $DATASET_DIR/mesh/"
    echo "Output infilled: $OUTPUT_FOLDER_INFILLED/"
    echo "Output GT:       $OUTPUT_FOLDER_GT/"
    echo "Resolution:      $RESOLUTION"
    echo "Samples/voxel:   $SAMPLES_PER_VOXEL"
    echo ""

    INFILL_START_TIME=$(date +%s)

    bash infill.sh \
        --input_folder "$DATASET_DIR" \
        --output_folder_infilled "$OUTPUT_FOLDER_INFILLED" \
        --output_folder_gt "$OUTPUT_FOLDER_GT" \
        --resolution "$RESOLUTION" \
        --samples_per_voxel "$SAMPLES_PER_VOXEL"

    INFILL_END_TIME=$(date +%s)
    INFILL_DURATION=$((INFILL_END_TIME - INFILL_START_TIME))

    echo ""
    echo "Mesh infilling complete! (Time: ${INFILL_DURATION}s)"
    echo ""
fi

# =============================================================================
# STEP 4: Inference + Validation (Both Modes)
# =============================================================================

# Validate that required data exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory '$DATASET_DIR' not found!"
    exit 1
fi

if [ ! -d "$SKELETON_DIR" ]; then
    echo "ERROR: Skeleton directory '$SKELETON_DIR' not found!"
    echo "  Run full pipeline first to generate skeleton data from FBX."
    exit 1
fi

if [ ! -d "${DATASET_DIR}/infilled" ]; then
    echo "ERROR: Infilled directory '${DATASET_DIR}/infilled' not found!"
    echo "  Run full pipeline first to generate infilled data from FBX."
    exit 1
fi

echo ""
echo "============================================================"
echo "STEP 4a: Running Inference to Generate Ground Truth"
echo "============================================================"
echo ""

INFERENCE_SCRIPT="exp_motion/train/gt_code/baseline_gt_tjx.py"
INFERENCE_OUTPUT_DIR="output/inference/${DATASET}"
WANDB_NAME_INF="${DATASET}_inference"

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

# Clean old inference outputs
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
    --position_method $POSITION_METHOD \
    --cuboid_size_mode $CUBOID_SIZE_MODE \
    --cuboid_size_coeff $CUBOID_SIZE_COEFF \
    --cuboid_knn_k $CUBOID_KNN_K

INFERENCE_END_TIME=$(date +%s)
INFERENCE_DURATION=$((INFERENCE_END_TIME - INFERENCE_START_TIME))

echo ""
echo "Inference complete! (Time: ${INFERENCE_DURATION}s)"
echo ""

# =============================================================================
# STEP 4b: Copy Inference Output to GT Directory
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 4b: Setting Up Ground Truth from Inference Output"
echo "============================================================"
echo ""

# Find the newly created inference output subdirectory
INFERENCE_SUBDIR=$(ls -td "$INFERENCE_OUTPUT_DIR"/*/ 2>/dev/null | head -1)

if [ -z "$INFERENCE_SUBDIR" ]; then
    echo "ERROR: No inference output found in '$INFERENCE_OUTPUT_DIR'"
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
    echo "Copied gt_*.ply files to $GT_DIR"
else
    echo "WARNING: No gt_*.ply files found in inference output."
    echo "  Checking for existing GT files in dataset..."

    if ls "${DATASET_DIR}/output"/gt_*.ply 1>/dev/null 2>&1; then
        echo "  Using existing GT files from ${DATASET_DIR}/output/"
        cp "${DATASET_DIR}/output"/gt_*.ply "$GT_DIR/"
    else
        echo "ERROR: No GT files available!"
        exit 1
    fi
fi

# Count GT frames
NUM_GT_FILES=$(ls -1 "$GT_DIR"/gt_*.ply 2>/dev/null | wc -l)
echo "Total GT frames available: $NUM_GT_FILES"
echo ""

# =============================================================================
# STEP 4c: Run Validation (Training)
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 4c: Running Validation Against Inference Output"
echo "============================================================"
echo ""

TRAIN_SCRIPT="exp_motion/train/spv_code/baseline_spv_bm_pu_tjx_SGD.py"
TRAIN_OUTPUT_DIR="output/train/${DATASET}"
WANDB_NAME_TRAIN="${DATASET}_validation"

# Validation parameters
NUM_FRAMES_TRAIN=$NUM_GT_FILES
SUBSTEP_TRAIN=$SUBSTEP_INF
YOUNGS_TRAIN=$YOUNGS_INF
NU_TRAIN=$NU_INF
SAMPLE_PARTICLES_TRAIN=$SAMPLE_PARTICLES_INF
VELO_FACTOR_TRAIN=0.0

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
    --cuboid_size_mode $CUBOID_SIZE_MODE \
    --cuboid_size_coeff $CUBOID_SIZE_COEFF \
    --cuboid_knn_k $CUBOID_KNN_K

VALIDATION_END_TIME=$(date +%s)
VALIDATION_DURATION=$((VALIDATION_END_TIME - VALIDATION_START_TIME))

echo ""
echo "Validation complete! (Time: ${VALIDATION_DURATION}s)"
echo ""

# =============================================================================
# STEP 4d: Generate Cuboid Sphere Meshes
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 4d: Generating Cuboid Sphere Meshes"
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
        echo "Cuboid meshes generated in: $CUBOID_OUTPUT_DIR"
    else
        echo "WARNING: Could not find cuboid_sizes.npy or skeleton directory"
        echo "  Skipping cuboid mesh generation"
    fi
else
    echo "WARNING: Could not find training output directory"
    echo "  Skipping cuboid mesh generation"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

# Calculate total time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "TIMING:"
if [ "$RUN_MODE" = "full" ]; then
    echo "  - FBX Conversion:  ${FBX_DURATION}s"
    echo "  - Skeleton Points: ${SKEL_DURATION}s"
    echo "  - Mesh Infilling:  ${INFILL_DURATION}s"
fi
echo "  - Inference:       ${INFERENCE_DURATION}s"
echo "  - Validation:      ${VALIDATION_DURATION}s"
echo "  - TOTAL:           ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "Dataset:  $DATASET_DIR"
echo "Results:  $TRAIN_OUTPUT_DIR"
echo ""

# Loss statistics
FRAME_LOSS_FILE=$(ls -t "${TRAIN_OUTPUT_DIR}"/*/frame_losses.txt 2>/dev/null | head -1)

if [ -f "$FRAME_LOSS_FILE" ]; then
    echo "============================================================"
    echo "Training Loss Statistics"
    echo "============================================================"

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
echo ""
