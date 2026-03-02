#!/usr/bin/env bash
set -euo pipefail

# Example:
#   ./retarget.sh data=alligator
#   ./retarget.sh data=mixamo_idle fps=30
#   ./retarget.sh data=mixamo_idle results_dir=Results/mixamo_idle/mixamo_idle

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_NAME=""
SOURCE_FBX=""
RESULTS_DIR=""
OUT_GLB=""
MESH_OBJ=""
FPS_VAL="30"
OBJ_SCALE="auto"
COORD_SPACE="auto"
SAVE_BLEND="0"

for arg in "$@"; do
  case "$arg" in
    data=*) DATA_NAME="${arg#data=}" ;;
    source_fbx=*) SOURCE_FBX="${arg#source_fbx=}" ;;
    results_dir=*) RESULTS_DIR="${arg#results_dir=}" ;;
    out=*) OUT_GLB="${arg#out=}" ;;
    mesh_obj=*) MESH_OBJ="${arg#mesh_obj=}" ;;
    fps=*) FPS_VAL="${arg#fps=}" ;;
    obj_scale=*) OBJ_SCALE="${arg#obj_scale=}" ;;
    coord_space=*) COORD_SPACE="${arg#coord_space=}" ;;
    save_blend=*) SAVE_BLEND="${arg#save_blend=}" ;;
    *)
      echo "Unknown argument: $arg"
      echo "Supported: data= source_fbx= results_dir= out= mesh_obj= fps= obj_scale= coord_space= save_blend="
      exit 1
      ;;
  esac
done

if [[ -z "$DATA_NAME" ]]; then
  echo "Missing required argument: data=<name>"
  exit 1
fi

BLENDER_BIN="${BLENDER_BIN:-blender}"
if ! command -v "$BLENDER_BIN" >/dev/null 2>&1; then
  if [[ "$BLENDER_BIN" == "blender" && -x "/Applications/Blender.app/Contents/MacOS/Blender" ]]; then
    BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
  else
    echo "Blender executable not found: $BLENDER_BIN"
    echo "Set BLENDER_BIN to the Blender executable path."
    exit 1
  fi
fi

has_obj_seq() {
  local dir="$1"
  [[ -d "$dir" ]] && find "$dir" -maxdepth 1 -type f -name 'mesh_frame_*.obj' | grep -q .
}

if [[ -z "$SOURCE_FBX" ]]; then
  SOURCE_FBX="$SCRIPT_DIR/Data/fbx/$DATA_NAME.fbx"
elif [[ "$SOURCE_FBX" != /* ]]; then
  SOURCE_FBX="$SCRIPT_DIR/$SOURCE_FBX"
fi

if [[ ! -f "$SOURCE_FBX" ]]; then
  echo "source_fbx not found: $SOURCE_FBX"
  echo "Expected default: Data/fbx/$DATA_NAME.fbx"
  exit 1
fi

if [[ -z "$RESULTS_DIR" ]]; then
  candidates=(
    "$SCRIPT_DIR/Results/$DATA_NAME"
    "$SCRIPT_DIR/Results/$DATA_NAME/$DATA_NAME"
  )
  RESULTS_DIR=""
  for c in "${candidates[@]}"; do
    if has_obj_seq "$c"; then
      RESULTS_DIR="$c"
      break
    fi
  done
else
  if [[ "$RESULTS_DIR" != /* ]]; then
    RESULTS_DIR="$SCRIPT_DIR/$RESULTS_DIR"
  fi
fi

if [[ -z "$RESULTS_DIR" ]]; then
  echo "Could not auto-find mesh OBJ sequence for data=$DATA_NAME"
  echo "Tried:"
  echo "  Results/$DATA_NAME"
  echo "  Results/$DATA_NAME/$DATA_NAME"
  echo "Pass results_dir=... explicitly."
  exit 1
fi

if ! has_obj_seq "$RESULTS_DIR"; then
  echo "No mesh_frame_*.obj files found in: $RESULTS_DIR"
  exit 1
fi

if [[ -z "$OUT_GLB" ]]; then
  OUT_GLB="$SCRIPT_DIR/Results/${DATA_NAME}.glb"
elif [[ "$OUT_GLB" != /* ]]; then
  OUT_GLB="$SCRIPT_DIR/$OUT_GLB"
fi
mkdir -p "$(dirname "$OUT_GLB")"

if [[ "$COORD_SPACE" != "auto" && "$COORD_SPACE" != "world" && "$COORD_SPACE" != "local" ]]; then
  echo "coord_space must be one of: auto, world, local"
  exit 1
fi

if [[ "$SAVE_BLEND" != "0" && "$SAVE_BLEND" != "1" ]]; then
  echo "save_blend must be 0 or 1"
  exit 1
fi

cmd=(
  "$BLENDER_BIN" --background
  --python "$SCRIPT_DIR/retarget_results_to_textured.py"
  --
  --source_fbx "$SOURCE_FBX"
  --obj_seq_dir "$RESULTS_DIR"
  --output_glb "$OUT_GLB"
  --fps "$FPS_VAL"
  --obj_scale "$OBJ_SCALE"
  --coord_space "$COORD_SPACE"
)

if [[ -n "$MESH_OBJ" ]]; then
  cmd+=(--mesh_obj "$MESH_OBJ")
fi
if [[ "$SAVE_BLEND" == "1" ]]; then
  cmd+=(--save_blend)
fi

echo "data:        $DATA_NAME"
echo "source_fbx:  $SOURCE_FBX"
echo "obj_seq_dir: $RESULTS_DIR"
echo "output_glb:  $OUT_GLB"
echo "fps:         $FPS_VAL"
echo "obj_scale:   $OBJ_SCALE"
echo "coord_space: $COORD_SPACE"
echo "save_blend:  $SAVE_BLEND"
echo
"${cmd[@]}"

echo
echo "Done. Wrote: $OUT_GLB"
