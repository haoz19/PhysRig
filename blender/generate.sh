#!/usr/bin/env bash
set -euo pipefail

# Optional CLI override:
#   ./generate.sh data=walk.fbx
#   ./generate.sh data=subdir/mixamo_walk.fbx
DATA_ARG=""
for arg in "$@"; do
  case "$arg" in
    data=*)
      DATA_ARG="${arg#data=}"
      ;;
  esac
done

# Set this to the folder name inside Truebones/ that you want to process.
# Example: DATASET_NAME="Elephant"
DATASET_NAME=""

# Optional subset under the dataset.
# Example: SUBSET_NAME="SubsetA"
SUBSET_NAME=""

# Optional single FBX path relative to Truebones/$DATASET_NAME.
# Example: FBX_REL_PATH="SubsetA/walk.fbx" or "walk.fbx"
FBX_REL_PATH="trex.fbx"

# Blender executable (change if needed).
BLENDER_BIN="${BLENDER_BIN:-blender}"

# Optional object names expected in FBX files.
MESH_OBJ_NAME="${MESH_OBJ_NAME:-U3DMesh}"
SKELETON_OBJ_NAME="${SKELETON_OBJ_NAME:-Armature}"
SCALE_FACTOR="${SCALE_FACTOR:-15}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FBX_FALLBACK_ROOT="$SCRIPT_DIR/Data/fbx"
#INPUT_DIR="$SCRIPT_DIR/Truebones/$DATASET_NAME"
INPUT_DIR="$SCRIPT_DIR/$DATASET_NAME"
OUTPUT_ROOT="$SCRIPT_DIR/Data/$DATASET_NAME"

if [[ -n "$DATA_ARG" ]]; then
  # `data=` paths are script-dir relative, e.g. data=mixamo_walk.fbx
  INPUT_DIR="$SCRIPT_DIR"
  FBX_REL_PATH="$DATA_ARG"
  OUTPUT_ROOT="$SCRIPT_DIR/Data"
fi

if [[ ! -d "$INPUT_DIR" && ! -d "$FBX_FALLBACK_ROOT" ]]; then
  echo "Input directory not found: $INPUT_DIR"
  echo "FBX fallback directory not found: $FBX_FALLBACK_ROOT"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

if ! command -v "$BLENDER_BIN" >/dev/null 2>&1; then
  if [[ "$BLENDER_BIN" == "blender" && -x "/Applications/Blender.app/Contents/MacOS/Blender" ]]; then
    BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
  else
    echo "Blender executable not found: $BLENDER_BIN"
    echo "Set BLENDER_BIN to the full executable path and retry."
    exit 1
  fi
fi

SCAN_ROOT="$INPUT_DIR"
if [[ -n "$SUBSET_NAME" ]]; then
  SCAN_ROOT="$INPUT_DIR/$SUBSET_NAME"
fi
FALLBACK_SCAN_ROOT="$FBX_FALLBACK_ROOT"
if [[ -n "$DATASET_NAME" ]]; then
  FALLBACK_SCAN_ROOT="$FALLBACK_SCAN_ROOT/$DATASET_NAME"
fi
if [[ -n "$SUBSET_NAME" ]]; then
  FALLBACK_SCAN_ROOT="$FALLBACK_SCAN_ROOT/$SUBSET_NAME"
fi
FALLBACK_DATASET_ROOT="$FBX_FALLBACK_ROOT"
if [[ -n "$DATASET_NAME" ]]; then
  FALLBACK_DATASET_ROOT="$FBX_FALLBACK_ROOT/$DATASET_NAME"
fi

fbx_files=()
if [[ -n "$FBX_REL_PATH" ]]; then
  one_fbx=""
  candidate_paths=(
    "$INPUT_DIR/$FBX_REL_PATH"
    "$FBX_FALLBACK_ROOT/$FBX_REL_PATH"
  )
  if [[ -n "$DATASET_NAME" ]]; then
    candidate_paths+=("$FALLBACK_DATASET_ROOT/$FBX_REL_PATH")
  fi
  for candidate in "${candidate_paths[@]}"; do
    if [[ -f "$candidate" ]]; then
      one_fbx="$candidate"
      break
    fi
  done
  if [[ -z "$one_fbx" ]]; then
    echo "FBX file not found. Tried:"
    for candidate in "${candidate_paths[@]}"; do
      echo "  $candidate"
    done
    exit 1
  fi
  fbx_files+=("$one_fbx")
else
  scan_roots=()
  if [[ -d "$SCAN_ROOT" ]]; then
    scan_roots+=("$SCAN_ROOT")
  fi
  if [[ -d "$FALLBACK_SCAN_ROOT" && "$FALLBACK_SCAN_ROOT" != "$SCAN_ROOT" ]]; then
    scan_roots+=("$FALLBACK_SCAN_ROOT")
  fi

  if [[ ${#scan_roots[@]} -eq 0 ]]; then
    echo "No scan roots found."
    echo "Tried:"
    echo "  $SCAN_ROOT"
    echo "  $FALLBACK_SCAN_ROOT"
    exit 1
  fi

  while IFS= read -r path; do
    fbx_files+=("$path")
  done < <(find "${scan_roots[@]}" -type f \( -iname "*.fbx" \) | sort -u)
fi

if [[ ${#fbx_files[@]} -eq 0 ]]; then
  echo "No FBX files found."
  echo "Scanned:"
  if [[ -d "$SCAN_ROOT" ]]; then
    echo "  $SCAN_ROOT"
  fi
  if [[ -d "$FALLBACK_SCAN_ROOT" ]]; then
    echo "  $FALLBACK_SCAN_ROOT"
  fi
  exit 1
fi

echo "Dataset:      $DATASET_NAME"
echo "Input dir:    $INPUT_DIR"
echo "Scan root:    $SCAN_ROOT"
echo "Fallback fbx: $FBX_FALLBACK_ROOT"
echo "Output root:  $OUTPUT_ROOT"
echo "Blender bin:  $BLENDER_BIN"
echo "Scale factor: ${SCALE_FACTOR}x"
echo "FBX count:    ${#fbx_files[@]}"
echo

for fbx_path in "${fbx_files[@]}"; do
  fbx_name="$(basename "$fbx_path")"
  fbx_name_lc="$(printf '%s' "$fbx_name" | tr '[:upper:]' '[:lower:]')"
  clip_name="${fbx_name%.*}"
  if [[ "$fbx_path" == "$INPUT_DIR/"* ]]; then
    rel_path="${fbx_path#$INPUT_DIR/}"
  elif [[ "$fbx_path" == "$FBX_FALLBACK_ROOT/"* ]]; then
    rel_path="${fbx_path#$FBX_FALLBACK_ROOT/}"
  else
    rel_path="$fbx_name"
  fi
  rel_dir="$(dirname "$rel_path")"
  if [[ "$rel_dir" == "." ]]; then
    clip_out="$OUTPUT_ROOT/$clip_name"
  else
    clip_out="$OUTPUT_ROOT/$rel_dir/$clip_name"
  fi

  echo "Converting: $rel_path"
  echo " -> $clip_out"
  if [[ "$fbx_name_lc" == mixamo* ]]; then
    converter_script="$SCRIPT_DIR/fbx_to_pipeline_mixamo.py"
    echo " -> converter: mixamo"
    "$BLENDER_BIN" --background --python "$converter_script" -- \
      "$fbx_path" "$clip_out" \
      --scale "$SCALE_FACTOR"
  else
    converter_script="$SCRIPT_DIR/fbx_to_pipeline_truebones.py"
    echo " -> converter: truebones"
    "$BLENDER_BIN" --background --python "$converter_script" -- \
      "$fbx_path" "$clip_out" \
      --mesh_obj "$MESH_OBJ_NAME" \
      --skeleton_obj "$SKELETON_OBJ_NAME" \
      --scale "$SCALE_FACTOR"
  fi
done

echo
echo "Done. Outputs are under: $OUTPUT_ROOT"
