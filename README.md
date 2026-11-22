# CoCO-Train (Compact CUDA OpenGL – Training Pipeline)

Renderer-aware training pipeline for **3D Gaussian Splatting (3DGS)**, forked from the official [Inria 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository (`3dgs_accel` branch) and stripped of the legacy SIBR viewer in favor of [CoCO-Splat](https://github.com/RareIcubu/CoCO-Splat).

Designed to produce compact, low-VRAM Gaussian scenes optimized for real-time rendering on consumer GPUs.

## Key Modifications (vs. Vanilla 3DGS)

- **Opacity-Aware Tight AABB (`--tight_aabb`):** Propagated to the custom [CoCO-DGR](https://github.com/RareIcubu/CoCO-DGR) CUDA rasterizer via `COCO_TIGHT_AABB=1`, replacing loose square bounds with opacity-scaled rectangular screen-space bounds to reduce false-positive tile intersections.
- **Opacity Entropy Regularization (`--opacity_entropy`):** Adds a smooth quadratic binary-pushing penalty $4 \cdot o \cdot (1 - o)$ weighted by `--lambda_opacity` (scheduled from `densify_from_iter` or `densify_until_iter`) to drive semi-transparent floaters toward opaque or prunable states.
- **Importance-Based Prune & Recovery (`--prune_and_recovery`):** Accumulates per-Gaussian importance scores between `densify_until_iter` and `--prune_iteration`, removes dead splats ($o < 1/255$) along with the lowest-importance fraction (`--prune_percent`), and recovers quality during the remaining iterations.
- **CoCO-Splat Evaluation Export (`render.py`):** Exports test camera trajectories (`test_cameras.txt`) compatible with `CoCO-Splat`'s interactive (`F5`) and headless `screenshot` evaluation modes.
- **VRAM Telemetry & Ablation Runner:** Logs live and peak allocated/reserved GPU memory (`training_stats.txt`) and provides `experiment_runner.py` (using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) for automated ablation studies.

## Environment & Installation

> **Note:** Tested with PyTorch `2.14.1+cu130` (CUDA 13.0). Make sure the installed PyTorch CUDA version matches your local `nvcc` (`CUDA Toolkit`) before compiling the CUDA extensions below.

```bash
# 1. Clone recursively (includes CoCO-DGR and helper submodules)
git clone --recursive [https://github.com/RareIcubu/CoCO-Train.git](https://github.com/RareIcubu/CoCO-Train.git)
cd CoCO-Train

# 2. Create and activate the Conda environment
conda env create --file environment.yml
conda activate gaussian_splatting

# 3. Compile CUDA extensions (without build isolation so setup.py sees PyTorch)
pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
# Uncomment if fused-ssim is present in submodules/:
# pip install --no-build-isolation submodules/fused-ssim
``````

## Custom CLI Arguments

All standard Inria 3DGS arguments remain supported. Additional parameters introduced in `CoCO-Train`:

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--tight_aabb` | `False` | Enables opacity-aware tight rectangular AABBs in `CoCO-DGR` (training and rendering). |
| `--opacity_entropy` | `False` | Enables quadratic opacity entropy regularization. |
| `--opacity_entropy_start` | `densify_until_iter` | Start point for opacity regularization: `densify_from_iter` or `densify_until_iter`. |
| `--lambda_opacity` | `0.0001` | Weight of the opacity regularization term (ablation target: `0.005`). |
| `--prune_and_recovery` | `False` | Enables post-densification importance pruning and recovery. |
| `--prune_iteration` | `20000` | Iteration at which pruning runs (`densify_until_iter < iter <= iterations`). |
| `--prune_percent` | `0.20` | Fraction of lowest-importance Gaussians removed during pruning. |

### Single Training Run

```bash
python train.py \
  -s /path/to/colmap_scene \
  -m ./output/scene_coco \
  --eval \
  --iterations 30000 \
  --tight_aabb \
  --opacity_entropy \
  --opacity_entropy_start densify_until_iter \
  --lambda_opacity 0.005 \
  --prune_and_recovery \
  --prune_iteration 20000 \
  --prune_percent 0.20
```

### Rendering & Metrics

```bash
python render.py -m ./output/scene_coco --iteration 30000 --skip_train
python metrics.py -m ./output/scene_coco
```

### Automated Ablation Study

```bash
# Run full 4-variant ablation with tight AABB enabled
python experiment_runner.py \
  -s /path/to/colmap_scene \
  -o ./output/scene_ablation \
  --tight_aabb

# Run only the target variant
python experiment_runner.py \
  -s /path/to/colmap_scene \
  -o ./output/scene_ablation \
  --tight_aabb \
  --only oe_late_pr
```

## Attribution & References

Original codebase and license terms are retained in [LICENSE.md](LICENSE.md). This pipeline builds on and draws algorithmic ideas from:
- **3D Gaussian Splatting:** Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG 2023.
- **Taming 3DGS (`3dgs_accel`):** Mallick et al., *Taming 3DGS: High-Quality Radiance Fields with Limited Resources*, SIGGRAPH Asia 2024.
- **Opacity-Aware Tight AABBs:** Radl et al. (*StopThePop*, ACM TOG 2024), Hanson et al. (*Speedy-Splat*, CVPR 2025), and Hahlbohm et al. (*Faster-GS*, 2026).
- **Importance-Based Prune & Recovery:** Fan et al., *LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS*, 2024.
