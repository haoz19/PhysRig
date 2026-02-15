#!/bin/bash

set -e

INPUT_FOLDER="data/dragon_tjx"
OUTPUT_FOLDER="data/dragon_tjx/skeleton_test"
NUM_FRAMES=80

if [ ! -d "$INPUT_FOLDER/skeleton_mesh" ]; then
    echo "Error: Missing $INPUT_FOLDER/skeleton_mesh"
    exit 1
fi

mkdir -p "$OUTPUT_FOLDER"

mkdir -p "$OUTPUT_FOLDER/skeleton"
rm -f "$OUTPUT_FOLDER/skeleton"/gt_*.ply

python exp_motion/utils/generate_skeleton_centroids.py \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER/skeleton" \
    --num_frames "$NUM_FRAMES"

echo "Skeleton centroids written to $OUTPUT_FOLDER/skeleton"
