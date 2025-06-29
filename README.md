# 🔩 PhysRig: Differentiable Physics-Based Rigging for Realistic Articulated Object Modeling

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2506.20936-b31b1b.svg)](https://arxiv.org/abs/2506.20936)
[![Project Page](https://img.shields.io/badge/Project-physrig.github.io-blue)](https://physrig.github.io)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-red)](https://iccv2025.thecvf.com/)

PhysRig is a differentiable, physics-based skinning and rigging framework that enables **realistic deformation** of articulated 3D objects. Unlike traditional methods like Linear Blend Skinning (LBS), PhysRig embeds skeletons into a deformable soft-body volume simulated using **Material Point Method (MPM)** — capturing the behavior of soft tissues, tails, ears, and other elastic structures in a physically plausible way.

> 🔧 From UIUC & Stability AI | 🦖 ICCV 2025 Accepted

---

## 🧠 Highlights

- **Differentiable Physics-Based Simulation**  
  Deformations are modeled using MPM and continuum mechanics, supporting gradient-based optimization of both motion and material properties.

- **Material Prototypes**  
  Learnable elastic parameter templates (Young’s modulus & Poisson’s ratio) provide expressive yet compact material modeling.

- **Driving Point Rigging**  
  Instead of bone transformations, PhysRig uses velocity-driven embedded points to induce dynamic shape changes.

- **Strong Performance**  
  Outperforms LBS on both user studies and geometric metrics (Chamfer Distance), with better convergence and physical realism.

- **Applications**  
  - Inverse Skinning (from motion to parameters)  
  - Pose Transfer (e.g., human → jellyfish)  
  - 4D Reconstruction & Animation

---

## Code will be released to the public around ICCV
