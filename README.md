# Benchmarking Lion Against Standard Optimizers Across Multiple Tasks

A clean, reproducible deep learning benchmark for comparing optimizers across vision workloads, including **semantic segmentation** with **DeepLabV3-ResNet50** on **Pascal VOC 2012** and an experimental **3D Half-Gaussian Splatting (3DHGS)** neural-rendering pipeline.

This repository was built as part of a graduate-level empirical study on optimizer behavior across tasks. The current phase focuses on a vision benchmark and compares **SGD**, **Adam**, **AdamW**, and **Lion** under a fixed training budget while tracking both model quality and systems efficiency.

## Why this project matters

Most optimizer comparisons are presented only through final accuracy. This project goes further by measuring:

- segmentation quality through `mIoU`, loss, and pixel accuracy
- convergence behavior across epochs
- optimizer update cost through per-step timing
- memory footprint through optimizer state tracking
- reproducibility through YAML-driven experiments and saved histories

The broader project roadmap extends this benchmark to **ADE20K**, **AG News**, and scene-level neural rendering so the final study covers multiple training regimes rather than a single easy classification benchmark.

## Current benchmarks

- **Segmentation task:** Pascal VOC 2012 with DeepLabV3-ResNet50
- **Neural-rendering task:** 3DHGS scene optimization for novel-view synthesis
- **Optimizers:** SGD, Adam, AdamW, and Lion
- **Tracked systems metrics:** step time, optimizer-step time, optimizer-state memory, parameter memory, and CUDA peak memory
- **GPU used for Pascal VOC experiments:** NVIDIA RTX 3060 (6 GB)
- **Larger-scale target hardware:** V100, A100, or H100 when available

## Key results

Under the common 25-epoch comparison budget:

- **SGD** achieved the strongest overall performance with best validation `mIoU = 0.6564`
- **AdamW** was the strongest adaptive baseline with best validation `mIoU = 0.6421`
- **Adam** remained competitive but trailed SGD and AdamW
- **Lion** used lower optimizer-state memory than Adam/AdamW, but underperformed in both accuracy and optimizer-step efficiency in this setup

These results make Pascal VOC a strong first-stage benchmark for a larger optimizer study across datasets and tasks.

## Repository structure

```text
651_project/
|-- configs/                 # YAML experiment configs
|-- docs/                    # Report source and report-ready figures
|-- 3DHGS/                   # 3D Half-Gaussian Splatting optimizer instrumentation
|-- outputs/                 # Training runs, checkpoints, and comparison plots
|-- scripts/                 # Training, sanity checks, and plotting entry points
|-- src/
|   |-- data/                # Pascal VOC dataset and transforms
|   |-- models/              # Model factory for segmentation architectures
|   `-- training/            # Step logic, epoch metrics, and optimizer utilities
|-- README.md
|-- requirements.txt         # Main segmentation dependencies
`-- requirements-3dhgs.txt   # Extra notes/dependencies for 3DHGS
```

## Main features

- Pascal VOC dataloader with paired image-mask transforms
- DeepLabV3-ResNet50 model factory
- YAML-configured training pipeline with `tqdm` progress bars
- Epoch-level tracking for:
  - loss
  - `mIoU`
  - pixel accuracy
  - throughput
  - forward/backward/optimizer step time
  - optimizer-state memory
- Multi-optimizer experiment setup
- Plot generation for side-by-side comparison
- LaTeX report assets for submission-ready writeups

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using the same Windows-based environment from WSL, the commands used in this project looked like:

```bash
/mnt/c/Users/HIMANSHU/Desktop/langchain/practice/myenv/Scripts/python.exe -m pip install -r requirements.txt
```

## Dataset setup

Download Pascal VOC 2012 through the dataloader utility:

```bash
python scripts/check_voc_dataloader.py --download
```

By default, the dataset is stored under:

```text
data/VOCdevkit/VOC2012
```

## Training

The default SGD configuration is:

- learning rate: `0.01`
- momentum: `0.9`
- weight decay: `1e-4`

Start training with:

```bash
python scripts/train.py --config configs/deeplabv3_resnet50_voc.yaml
```

Run the other optimizers with:

```bash
python scripts/train.py --config configs/deeplabv3_resnet50_voc_adam_50ep.yaml
python scripts/train.py --config configs/deeplabv3_resnet50_voc_adamw_50ep.yaml
python scripts/train.py --config configs/deeplabv3_resnet50_voc_lion_50ep.yaml
```

If you are launching from WSL with the Windows interpreter used during development:

```bash
/mnt/c/Users/HIMANSHU/Desktop/langchain/practice/myenv/Scripts/python.exe scripts/train.py --config configs/deeplabv3_resnet50_voc.yaml
```


## 3DHGS neural-rendering benchmark

The `3DHGS/` directory contains the project-specific modifications for testing optimizer behavior on **3D Half-Gaussian Splatting**, a per-scene neural-rendering method. Unlike classification or segmentation, this task does not train one model that generalizes across many scenes. Instead, each run optimizes a scene-specific 3D representation from posed images, then evaluates how well that representation renders held-out or training camera views.

This makes the 3DHGS task useful for a different kind of optimizer comparison: it measures how optimizers behave when fitting explicit 3D scene parameters, including convergence speed, reconstruction quality, optimizer memory, and per-step update cost.

### Restoring the full 3DHGS source

This repository intentionally tracks only the project-specific 3DHGS modifications, not the full upstream 3DHGS codebase, datasets, CUDA build outputs, or trained scene artifacts. To run the 3DHGS pipeline on a fresh machine, first restore the upstream repository into `3DHGS/`, then keep this project's modified files on top of it.

```bash
cd /mnt/c/Users/HIMANSHU/Downloads/651_project

# If 3DHGS is missing or incomplete, restore the upstream source locally.
git clone --recursive https://github.com/lihaolin88/3DHGS.git 3DHGS_upstream

# Copy upstream files into the local working 3DHGS directory without deleting
# this project's modified train.py, arguments/__init__.py, and scene/gaussian_model.py.
rsync -a --ignore-existing 3DHGS_upstream/ 3DHGS/
```

After this step, `3DHGS/` should contain the upstream submodules plus this project's optimizer-selection and metric-tracking changes. The extra upstream files remain ignored by the outer project repository.

### 3DHGS dependency setup

3DHGS uses custom CUDA extensions and is tied to an older stack than the segmentation pipeline. Use a separate conda environment rather than the main `requirements.txt` environment.

```bash
cd /mnt/c/Users/HIMANSHU/Downloads/651_project/3DHGS

conda env create --file environment.yml
source /home/himanshu/anaconda3/etc/profile.d/conda.sh
conda activate half_gaussian_splatting
```

For WSL/CUDA builds, the following environment variables were required during development:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```

Install the CUDA extensions from inside `3DHGS/`:

```bash
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install -r ../requirements-3dhgs.txt
```

If CUDA libraries are not found at runtime, first confirm that the conda environment is active and that `/usr/lib/wsl/lib` and `$CONDA_PREFIX/lib` are present in `LD_LIBRARY_PATH`.

### 3DHGS dataset layout

Keep datasets inside `3DHGS/data/`. This folder is intentionally ignored by Git.

Expected layout for the Tanks and Temples `truck` scene:

```text
3DHGS/data/tandt/truck/
|-- images/
|-- sparse/
`-- ...
```

Other downloaded scenes can be kept similarly, for example:

```text
3DHGS/data/tandt/train/
3DHGS/data/db/drjohnson/
3DHGS/data/db/playroom/
```

### Running a 3DHGS optimizer sanity test

A small run can verify that training, optimizer selection, and metrics logging work before launching long experiments. For the `truck` scene, one epoch corresponds to one pass over the available training cameras.

```bash
cd /mnt/c/Users/HIMANSHU/Downloads/651_project/3DHGS
source /home/himanshu/anaconda3/etc/profile.d/conda.sh
conda activate half_gaussian_splatting

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

python train.py \
  -s data/tandt/truck \
  -m outputs/test_lion_1epoch \
  --epochs 1 \
  --optimizer lion \
  --save_iterations 251 \
  --checkpoint_iterations 251
```

Supported optimizer names in the modified 3DHGS pipeline are:

- `adam`
- `adamw`
- `sgd`
- `lion`

### 3DHGS metrics and outputs

Each 3DHGS training run writes outputs under the folder passed with `-m`, for example:

```text
3DHGS/outputs/test_lion_1epoch/
|-- cfg_args
|-- training_metrics.csv
|-- training_metrics.jsonl
|-- chkpnt251.pth
`-- point_cloud/
```

The comparison metrics are saved at epoch boundaries, not every iteration. The CSV/JSONL logs include:

- `mean_loss`
- `mean_l1_loss`
- `mean_train_psnr`
- `mean_step_time_ms`
- `mean_forward_time_ms`
- `mean_backward_time_ms`
- `mean_optimizer_time_ms`
- `optimizer_state_bytes`
- `parameter_bytes`
- `cuda_allocated_bytes`
- `cuda_reserved_bytes`
- `cuda_peak_allocated_bytes`
- `cuda_peak_reserved_bytes`

The key field for optimizer memory comparison is `optimizer_state_bytes`.

### Rendering and visualizing 3DHGS results

Render trained views with:

```bash
python render.py \
  -s data/tandt/truck \
  -m outputs/test_lion_1epoch \
  --iteration 251 \
  --skip_test
```

Rendered images are written under the run directory, usually in a path like:

```text
3DHGS/outputs/test_lion_1epoch/train/ours_251/renders/
```

The `.ply` files produced by Gaussian-splatting methods are not ordinary meshes. MeshLab may open them, but it is not the right tool for judging visual quality. Use rendered images, PSNR/SSIM/LPIPS-style metrics, or the SIBR/3DHGS viewer when available.

### Git policy for 3DHGS

Only the project-specific 3DHGS modifications should be pushed to this repository. The full upstream checkout, datasets, outputs, checkpoints, compiled CUDA files, and generated point clouds are ignored. The currently tracked 3DHGS files are intended to be:

- `3DHGS/train.py`
- `3DHGS/arguments/__init__.py`
- `3DHGS/scene/gaussian_model.py`

This keeps the repository lightweight while preserving the optimizer-selection and tracking changes needed for the project.

## Outputs

Each run writes results to its own folder under `outputs/`, including:

- `history.jsonl` with per-epoch metrics
- `best.pt` for the best validation checkpoint
- `last.pt` for the latest checkpoint

Current experiment folders:

- `outputs/deeplabv3_resnet50_voc_sgd_50ep`
- `outputs/deeplabv3_resnet50_voc_adam_50ep`
- `outputs/deeplabv3_resnet50_voc_adamw_50ep`
- `outputs/deeplabv3_resnet50_voc_lion_50ep`

## Plot generation

Generate comparison plots with:

```bash
python scripts/plot_optimizer_comparison.py --max-epoch 25 --output-dir outputs/comparison_plots_25ep
```

The generated plots are stored in:

- `outputs/comparison_plots_25ep`
- `docs/report_figures` for report-ready renamed copies

## Sanity-check scripts

The repository includes small utilities to verify individual pieces before running full experiments:

- `scripts/check_voc_dataloader.py`
- `scripts/check_model.py`
- `scripts/check_train_step.py`
- `scripts/check_epoch_metrics.py`

These are useful for debugging dataset, model, and metric issues before launching long runs.

## Report assets

The LaTeX writeup and report figure assets are stored in:

- `docs/optimizer_report.tex`
- `docs/report_figures/`

## Future work

The next phase of the project extends the benchmark to:

- **ADE20K** for a harder semantic segmentation benchmark
- **AG News** for a text classification benchmark

That extension is meant to test whether optimizer behavior remains consistent across datasets, modalities, and training regimes.

## Author

Developed by **Ananya Jha, Himanshu Ranjan, Mayank, and Sethupathy Raghunathan Venkatraman**.

