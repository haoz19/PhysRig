#!/bin/bash

# Mesh infilling script
# Usage: bash infill.sh [OPTIONS]
# Run from PhysRig directory
# Note: Infill is only generated for the first frame (frame 0).
#       GT point clouds are generated for all frames specified by --num_frames.

# Default parameters (customizable via command line arguments)
INPUT_FOLDER="data/dragon_tjx"           # Input folder containing mesh/ and skeleton_mesh/ subfolders
OUTPUT_FOLDER_INFILLED="data/dragon_tjx/infilled"  # Output folder for infilled point clouds
OUTPUT_FOLDER_GT="data/dragon_tjx/gt"   # Output folder for GT point clouds
NUM_FRAMES=""                            # Number of frames for GT generation (empty = auto-detect)
                                          # Default: auto-detect from input folder, or 80 for dragon_tjx
                                          # This determines how many GT point clouds (gt_0.ply, gt_1.ply, ...) are generated
                                          # Note: Infill is always only generated for frame 0, regardless of NUM_FRAMES
RESOLUTION=20                            # Voxel grid resolution
SAMPLES_PER_VOXEL=5                      # Samples per voxel
NUM_FRAMES_EXPLICIT=false                # Track if --num_frames was explicitly set

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --output_folder_infilled)
            OUTPUT_FOLDER_INFILLED="$2"
            shift 2
            ;;
        --output_folder_gt)
            OUTPUT_FOLDER_GT="$2"
            shift 2
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            NUM_FRAMES_EXPLICIT=true
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
        --help)
            echo "Usage: bash infill.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input_folder PATH              Input folder containing mesh/ and skeleton_mesh/ subfolders (default: data/dragon_tjx)"
            echo "  --output_folder_infilled PATH    Output folder for infilled point clouds (default: data/dragon_tjx/infilled)"
            echo "  --output_folder_gt PATH          Output folder for GT point clouds (default: data/dragon_tjx/gt)"
            echo "  --num_frames N                   Number of frames to process for GT generation"
            echo "                                   (default: auto-detect from input folder, or 80 if detection fails)"
            echo "                                   Note: Infill is only generated for frame 0"
            echo "  --resolution N                   Voxel grid resolution (default: 50)"
            echo "  --samples_per_voxel N            Samples per voxel (default: 5)"
            echo "  --help                           Show this help message"
            echo ""
            echo "Example:"
            echo "  bash infill.sh --input_folder data/dragon_tjx --num_frames 80 --resolution 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set Python path
PYTHON=python  # Change this to your python path if needed
PYFILE=exp_motion/utils/mesh_infill.py

# Get the script directory (PhysRig root) and add it to PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder does not exist: $INPUT_FOLDER"
    exit 1
fi

# Check if mesh and skeleton_mesh subfolders exist
if [ ! -d "$INPUT_FOLDER/mesh" ]; then
    echo "Error: Mesh folder does not exist: $INPUT_FOLDER/mesh"
    exit 1
fi

if [ ! -d "$INPUT_FOLDER/skeleton_mesh" ]; then
    echo "Error: Skeleton mesh folder does not exist: $INPUT_FOLDER/skeleton_mesh"
    exit 1
fi

# Auto-detect number of frames if not explicitly set
if [ "$NUM_FRAMES_EXPLICIT" = false ]; then
    FRAME_COUNT=$(ls "$INPUT_FOLDER/mesh"/*.obj 2>/dev/null | wc -l)
    if [ "$FRAME_COUNT" -gt 0 ]; then
        NUM_FRAMES=$FRAME_COUNT
    else
        # Fallback to default if auto-detection fails
        NUM_FRAMES=80
    fi
fi

# Print configuration
echo "=========================================="
echo "Mesh Infilling Configuration"
echo "=========================================="
echo "Input folder:           $INPUT_FOLDER"
echo "Output folder (infilled): $OUTPUT_FOLDER_INFILLED"
echo "Output folder (GT):     $OUTPUT_FOLDER_GT"
if [ "$NUM_FRAMES_EXPLICIT" = false ]; then
    echo "Number of frames (GT):  $NUM_FRAMES (auto-detected, infill only for frame 0)"
else
    echo "Number of frames (GT):  $NUM_FRAMES (explicitly set, infill only for frame 0)"
fi
echo "Resolution:             $RESOLUTION"
echo "Samples per voxel:      $SAMPLES_PER_VOXEL"
echo "=========================================="
echo ""

# Run mesh infilling script
echo "Running mesh infilling..."
$PYTHON $PYFILE \
    --input_folder "$INPUT_FOLDER" \
    --output_folder_infilled "$OUTPUT_FOLDER_INFILLED" \
    --output_folder_gt "$OUTPUT_FOLDER_GT" \
    --num_frames $NUM_FRAMES \
    --resolution $RESOLUTION \
    --samples_per_voxel $SAMPLES_PER_VOXEL

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Mesh infilling complete!"
    echo "Infilled point clouds saved to: $OUTPUT_FOLDER_INFILLED"
    echo "GT point clouds saved to: $OUTPUT_FOLDER_GT"
    echo "=========================================="
else
    echo ""
    echo "Error: Mesh infilling failed!"
    exit 1
fi


