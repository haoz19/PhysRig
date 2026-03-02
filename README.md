# PhysRig: Differentiable Physics-Based Rigging for Realistic Articulated Object Modeling

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2506.20936-b31b1b.svg)](https://arxiv.org/abs/2506.20936)
[![Project Page](https://img.shields.io/badge/Project-physrig.github.io-blue)](https://physrig.github.io)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-red)](https://iccv2025.thecvf.com/)

PhysRig is a differentiable, physics-based skinning and rigging framework that enables **realistic deformation** of articulated 3D objects. Unlike traditional methods like Linear Blend Skinning (LBS), PhysRig embeds skeletons into a deformable soft-body volume simulated using **Material Point Method (MPM)** — capturing the behavior of soft tissues, tails, ears, and other elastic structures in a physically plausible way.

---

## Environment Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU
- Conda (recommended)

### Installation

```bash
conda create -n physrig python=3.10
conda activate physrig

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Install the custom Gaussian rasterization and simple-knn extensions:

```bash
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
```

---

## Data Preparation

Each dataset lives under `data/<dataset_name>/` and requires the following structure:

```
data/<dataset_name>/
├── mesh/                    # OBJ mesh sequence (mesh_frame_0000.obj, ...)
├── skeleton_mesh/           # Skeleton OBJ sequence (skeleton_frame_0000.obj, ...)
├── skeleton/                # (auto-generated) Skeleton centroid PLYs
├── infilled/                # (auto-generated) Infilled point clouds
└── gt/                      # (auto-generated) Ground truth point clouds
```

Only `mesh/` and `skeleton_mesh/` need to be provided manually. The remaining directories are generated automatically by the pipeline.

### From FBX files (requires Blender)

FBX animations must first be converted to OBJ sequences using the scripts in `blender/`. **This step requires Blender to be installed and must be run on a local machine** -- cloud servers (AWS, GCP, etc.) typically do not have Blender available.

```bash
cd blender

# Mixamo FBX files (auto-detected by filename prefix "mixamo"):
bash generate.sh data=mixamo_walk.fbx

# Truebones / other FBX files:
MESH_OBJ_NAME="Body" SKELETON_OBJ_NAME="Armature" bash generate.sh data=animation.fbx

# Batch-process a dataset folder:
DATASET_NAME="Elephant" bash generate.sh
```

The converted OBJ sequences will be saved under `blender/Data/`. Copy the resulting `mesh/` and `skeleton_mesh/` folders into `data/<dataset_name>/` on your training server.

### Manual preparation

If you already have OBJ sequences in `mesh/` and `skeleton_mesh/`, no Blender step is needed -- `inference.sh` will auto-generate `skeleton/` and `infilled/` data on the first run.

---

## Scripts

There are three main scripts. All are run from the PhysRig root directory.

### 1. `infill.sh` -- Mesh Infilling

Generates volumetric infill point clouds from the mesh sequence. Infill is only generated for frame 0; ground truth point clouds are generated for all frames.

```bash
bash infill.sh --input_folder data/<dataset> \
    --output_folder_infilled data/<dataset>/infilled \
    --output_folder_gt data/<dataset>/gt \
    --resolution 20 \
    --samples_per_voxel 5
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input_folder` | `data/dragon_tjx` | Folder containing `mesh/` and `skeleton_mesh/` |
| `--output_folder_infilled` | `data/dragon_tjx/infilled` | Where to write infilled point clouds |
| `--output_folder_gt` | `data/dragon_tjx/gt` | Where to write GT point clouds |
| `--num_frames` | auto-detect | Number of GT frames to generate |
| `--resolution` | `20` | Voxel grid resolution |
| `--samples_per_voxel` | `5` | Point samples per voxel |

You usually don't need to run this manually -- `inference.sh` calls it automatically when `infilled/` is missing.

### 2. `inference.sh` -- Physics Inference

Runs the full inference pipeline: skeleton generation, mesh infilling (if needed), MPM simulation, and mesh deformation.

```bash
bash inference.sh
```

Edit the top of `inference.sh` to set your dataset and parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET` | `mixamo_walk` | Dataset name (must exist under `data/`) |
| `CUBOID_UPDATE_MODE` | `both` | How cuboid positions/velocities update: `none`, `velocity_only`, `location_only`, `both` |
| `POSITION_METHOD` | `adaptive` | Cuboid center computation: `mean`, `median`, `weighted`, `bbox`, `adaptive`, `pca`, `optimized` |
| `CUBOID_SIZE_MODE` | `fixed` | Cuboid sizing strategy: `fixed`, `adaptive`, `knn`, `hybrid`, `ceil_and_floor`, `raycast` |
| `CUBOID_SIZE_COEFF` | `0.1` | Scaling coefficient for cuboid radius |
| `SUBSTEP_INF` | `100` | Simulation substeps per frame |
| `YOUNGS_INF` | `6e4` | Young's modulus (material stiffness) |
| `NU_INF` | `0.3` | Poisson's ratio |
| `FORCE_REGENERATE` | `false` | Delete and regenerate skeleton/infilled data every run |

The pipeline runs three stages automatically:

1. **Skeleton Generation** -- Converts `skeleton_mesh/` OBJs into centroid PLY files (skipped if `skeleton/` already exists)
2. **Mesh Infill** -- Calls `infill.sh` to generate volumetric point clouds (skipped if `infilled/` already exists)
3. **Physics Inference** -- Runs MPM simulation driven by skeleton velocities, then applies deformations to the mesh

**Outputs:**

```
output/
├── inference/<dataset>/     # Raw inference PLY frames
└── animation/<dataset>/     # Deformed mesh OBJ sequence (final output)
```

### 3. `train.sh` -- Material Training

Trains material parameters (Young's modulus, Poisson's ratio) to match target deformations. The script first runs inference as a validation step, then optimizes material fields via gradient descent.

```bash
bash train.sh
```

Edit the top of `train.sh` to configure the dataset, cuboid settings, and training hyperparameters (learning rate, iterations, gradient clipping, etc.).

---

## Project Structure

```
PhysRig/
├── inference.sh                 # Inference pipeline script
├── train.sh                     # Training pipeline script
├── infill.sh                    # Mesh infill generation
├── requirements.txt             # Python dependencies
├── blender/                     # FBX conversion (requires Blender, run locally)
│   ├── generate.sh              # Batch FBX-to-OBJ converter
│   ├── fbx_to_pipeline_mixamo.py
│   └── fbx_to_pipeline_truebones.py
├── exp_motion/
│   ├── train/
│   │   ├── cuboid_utils.py      # Cuboid finding & velocity assignment
│   │   ├── interface.py         # MPM simulation interface
│   │   ├── local_utils.py       # Local utilities
│   │   ├── new_utils.py         # Utility functions
│   │   ├── new2_utils.py        # Additional utilities
│   │   ├── new3_utils.py        # Additional utilities
│   │   ├── config.yml           # Default configuration
│   │   ├── gt_code/
│   │   │   └── baseline_gt_tjx.py   # Inference entry point
│   │   └── spv_code/
│   │       └── baseline_spv_bm_pu_tjx_SGD.py  # Training entry point
│   └── utils/
│       ├── apply_deformation.py          # PLY -> deformed mesh
│       ├── generate_cuboid_meshes.py     # Cuboid mesh visualization
│       ├── generate_skeleton_centroids.py # Skeleton PLY generation
│       └── mesh_infill.py               # Volumetric mesh infilling
├── thirdparty_code/             # Warp MPM solver
└── motionrep/                   # Motion representation library
```

---

## Citation

```bibtex
@inproceedings{zhang2025physrig,
    title={{PhysRig}: Physics-Based Rigging for Realistic Articulated Object Modeling},
    author={Zhang, Tianyuan and others},
    booktitle={ICCV},
    year={2025}
}
```
