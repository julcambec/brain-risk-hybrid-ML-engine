# HPC Environment Setup Guide

This guide documents the systems-level setup required to run the **brainrisk** pipeline on a high-performance computing (HPC) cluster. It covers CUDA version management, PyTorch compatibility, conda environment creation, FreeSurfer configuration, distributed training (DDP), and common pitfalls encountered during the original experiments on UBC Sockeye (NVIDIA V100 32 GB GPUs, SLURM scheduler).

The principles are broadly applicable to any SLURM-based HPC cluster with NVIDIA GPUs. Adapt module names and paths to your site.

---

## 1. CUDA and PyTorch Compatibility

PyTorch ships with a bundled CUDA runtime, but the cluster's CUDA **driver** (and sometimes toolkit modules) must be compatible with the version PyTorch was compiled against. A version mismatch could be a source of `CUDA error: no kernel image is available` or silent hangs during DDP initialisation.

### Version matrix (tested configurations)

| PyTorch | pytorch-cuda channel | Cluster CUDA module | GPU arch | Status |
|---------|---------------------|---------------------|----------|--------|
| 2.1.x   | 11.8                | cuda/11.8.0         | V100 (sm_70) | Stable: used for all MSc experiments |
| 2.2.x   | 11.8                | cuda/11.8.0         | V100 (sm_70) | Stable |
| 2.3.x   | 12.1                | cuda/12.1.0         | V100 / A100  | Tested on A100; V100 requires driver ≥ 525 |
| 2.4.x+  | 12.4                | cuda/12.4.0         | A100 / H100  | Not tested on V100 |

### How to check compatibility

```bash
# Check the cluster's CUDA driver version (must be ≥ toolkit version).
nvidia-smi | head -4

# After activating your conda env, verify PyTorch sees the GPUs.
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
"

# Verify NCCL (required for multi-GPU DDP).
python -c "import torch; print('NCCL:', torch.cuda.nccl.version())"
```

### Rules of thumb

- Match the `pytorch-cuda` conda channel version to the cluster CUDA module version, not the driver version. The driver just needs to be equal or newer.
- When in doubt, use CUDA 11.8: it has broad GPU architecture support (sm_37 through sm_90) and is battle-tested on old clusters.
- If you see `RuntimeError: CUDA error: no kernel image is available for execution on the device`, the PyTorch wheels were not compiled for your GPU's compute capability. Reinstall with the correct `pytorch-cuda` channel.

---

## 2. Conda Environment Setup

The brainrisk package uses Python ≥ 3.11 and separates dependencies into core, dev, and deep-learning groups (see `pyproject.toml`). On HPC, I'd use Miniforge (conda-forge default channel) for a lean base installation.

### Initial setup (one-time)

```bash
# Install Miniforge if not already available.
# Many clusters provide a module (e.g., `module load miniforge3`).
# Otherwise, install to your home directory:
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh

# Create the brainrisk environment.
conda create -n brainrisk python=3.12 -y
conda activate brainrisk

# Install PyTorch with CUDA support from the pytorch channel.
# Adjust pytorch-cuda to match your cluster (see version matrix above).
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install the brainrisk package in editable mode with dev + DL extras.
cd /path/to/brain-risk-hybrid-ML-engine
pip install -e ".[dev,dl]"
```

### Using the provided environment file

Alternatively, use the repo's `environment.yml` (installs only the core package; add PyTorch separately since the CUDA version is cluster-specific):

```bash
conda env create -f environment.yml
conda activate brainrisk
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Environment verification checklist

Run these checks after creating or updating the environment:

```bash
# Package is importable and CLI is registered.
brainrisk --help

# PyTorch + CUDA functional.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

# Quick forward-pass smoke test on a dummy volume.
python -c "
from brainrisk.deeplearning.model import MINiT
import torch
model = MINiT(volume_size=16, block_size=8, patch_size=4, embed_dim=32,
              num_layers=1, num_heads=4, mlp_dim=64, num_classes=3)
x = torch.randn(1, 1, 16, 16, 16)
print('Output shape:', model(x).shape)  # expect (1, 3)
print('Environment OK')
"
```

### Keeping environments reproducible

Pin exact versions for production runs:

```bash
# Export a fully resolved environment (useful for reproducing exact results).
conda env export --no-builds > environment_frozen.yml
# Or with pip:
pip freeze > requirements_frozen.txt
```

---

## 3. FreeSurfer Setup

FreeSurfer is required only for the **preprocessing shared core** (skull stripping, Talairach alignment, cortical parcellation). It is not needed for training or inference. The brainrisk demo mode bypasses FreeSurfer entirely using synthetic data.

### Loading FreeSurfer on HPC

Most neuroimaging clusters provide FreeSurfer as a module or a shared installation:

```bash
# Option A: cluster module (preferred).
module load freesurfer/7.4.1

# Option B: custom installation.
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

After loading, verify:

```bash
recon-all --version
mri_convert --version

# The brainrisk code also checks programmatically:
python -c "from brainrisk.preprocessing.freesurfer import is_freesurfer_available; print(is_freesurfer_available())"
```

### FreeSurfer and the brainrisk pipeline

The preprocessing pipeline (`brainrisk.preprocessing.pipeline`) calls FreeSurfer through the wrapper in `brainrisk.preprocessing.freesurfer`:

1. `recon-all -autorecon1` — motion correction, Talairach alignment, bias-field correction, skull stripping. Produces `brainmask.mgz` and `talairach.xfm.lta`.
2. `mri_convert` — converts `brainmask.mgz` to NIfTI for the DL branch.
3. `mri_vol2vol` — applies the Talairach affine to warp into MNI305 space (handled by `brainrisk.preprocessing.mni305` in pure Python for the demo).

For full `recon-all -all` (cortical parcellation for ROI extraction), wall time is approximately 6–8 hours per subject on a single CPU core. Use SLURM array jobs to parallelise (see `slurm_template.sh` for an example).

### License

FreeSurfer requires a license file. Obtain one and place it at `$FREESURFER_HOME/license.txt` or set `$FS_LICENSE`.

---

## 4. Distributed Data-Parallel (DDP) Training

The brainrisk training loop supports single-GPU, single-node multi-GPU, and multi-node training via PyTorch's DistributedDataParallel (DDP). The implementation lives in two modules:

- **`src/brainrisk/deeplearning/utils.py`** — `setup_distributed()` initialises the process group by reading the `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` environment variables (set by `torchrun`). It selects NCCL when CUDA is available and falls back to Gloo for CPU-only execution.
- **`src/brainrisk/deeplearning/trainer.py`** — `run_training()` wraps the model in `DistributedDataParallel` when distributed mode is active, uses a `DistributedSampler` for the training set, and restricts logging/checkpointing to rank 0.

### Launching DDP with torchrun

`torchrun` (PyTorch ≥ 1.10) replaces the older `torch.distributed.launch` and handles environment variable injection, and fault tolerance:

```bash
# Single-node, 4 GPUs:
torchrun --standalone --nproc_per_node=4 \
    -m brainrisk demo-dl --config configs/vit_subtype_classification.yaml

# Multi-node, 2 nodes × 4 GPUs (launched via srun inside SLURM):
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m brainrisk demo-dl --config configs/vit_subtype_classification.yaml
```

### Environment variables set by torchrun

| Variable | Meaning | Used in |
|----------|---------|---------|
| `RANK` | Global rank (0 … world_size - 1) | `setup_distributed()` |
| `WORLD_SIZE` | Total number of processes | `setup_distributed()` |
| `LOCAL_RANK` | Rank within the current node (0 … GPUs/node - 1) | `get_device()`; selects `cuda:<local_rank>` |
| `MASTER_ADDR` | IP of the rank-0 node | Used internally by `init_process_group` |
| `MASTER_PORT` | Port for rendezvous | Used internally by `init_process_group` |

### Effective batch size

With DDP, each process runs on its own GPU with `batch_size` samples. The effective batch size is:

```
effective_batch = batch_size × world_size
```

For the original experiments, a per-GPU batch size of 16 across 4 V100s gave an effective batch of 64, which stabilised training and contributed to the best sex-classification result (~80% validation accuracy). Adjust the learning rate proportionally when changing the effective batch size.

### Communication backends

| Backend | When | Notes |
|---------|------|-------|
| **NCCL** | CUDA is available | Optimal for GPU-to-GPU communication. Default on GPU clusters. |
| **Gloo** | CPU-only | Used for laptop/CI demos (`make demo-dl-track`). Functional but slow. |

The brainrisk code selects the backend automatically in `setup_distributed()`:

```python
backend = "nccl" if torch.cuda.is_available() else "gloo"
```

### Sampler and epoch setting

`DistributedSampler` partitions the dataset across ranks. **Call `sampler.set_epoch(epoch)` at the start of every epoch** to ensure proper shuffling. The brainrisk trainer handles this automatically:

```python
if train_sampler is not None:
    train_sampler.set_epoch(epoch)
```

---

## 5. Checkpoint Handling

### Save/resume contract

Checkpoints are saved by rank 0 only (via `is_main_process(rank)` guard). The `save_checkpoint()` function in `trainer.py` stores:

- Model state dict (unwrapped from DDP if applicable)
- Optimizer state dict
- Scheduler state dict
- Current epoch number
- Best validation metric
- Full training config (for reproducibility)

### Resuming training

Set `resume_path` in the YAML config to resume from a checkpoint:

```yaml
# In your config YAML:
resume_path: artifacts/dl/checkpoints/checkpoint_best.pt
```

The trainer calls `load_checkpoint()` which restores the model, optimizer, and scheduler states and resumes from the next epoch.

### Checkpoint storage on HPC

On shared HPC filesystems:

- **Write checkpoints to scratch space** (`/scratch/...`), not your home directory. Scratch usually has higher I/O throughput and higher quotas.
- **Copy final checkpoints to persistent storage** after the job completes, since scratch is typically purged periodically.
- The default checkpoint directory (`artifacts/dl/checkpoints/`) can be overridden via the YAML config (`checkpoint.save_dir`).

---

## 6. Common Pitfalls and Solutions

### NCCL errors on CPU-only machines

**Symptom:** `RuntimeError: ProcessGroupNCCL is only supported with GPUs` when running `make demo-dl-track` on a laptop.

**Cause:** torchrun was launched on a machine without GPUs but NCCL was selected.

**Fix:** The brainrisk code already handles this — `setup_distributed()` selects Gloo when `torch.cuda.is_available()` returns False. For the demo, run without torchrun:

```bash
python -m brainrisk demo-dl --config configs/demo_dl.yaml
```

### NCCL timeout / hang during init_process_group

**Symptom:** Training hangs at startup with no error message, or eventually times out.

**Causes and fixes:**
- **Firewall blocking inter-node communication.** Ask your HPC admin about open port ranges. Set `NCCL_SOCKET_IFNAME` to the correct network interface (e.g., `export NCCL_SOCKET_IFNAME=eth0`).
- **InfiniBand issues.** If your cluster does not have InfiniBand or it is misconfigured, disable it: `export NCCL_IB_DISABLE=1`.
- **Stale processes from a previous job.** Kill any lingering Python processes and retry.
- **Mismatched NCCL versions.** Ensure all nodes load the same NCCL version.

### cuDNN non-determinism

**Symptom:** Results vary across runs even with fixed seeds.

**Cause:** cuDNN autotuner selects different algorithms depending on hardware state.

**Fix:** The brainrisk `set_seed()` function in `deeplearning/utils.py` sets:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This sacrifices some speed for reproducibility. For production training where speed matters more than exact reproducibility, you can set `benchmark = True` in the config.

### OOM (out-of-memory) on GPU

**Symptom:** `RuntimeError: CUDA out of memory`.

**Fixes (in order of preference):**
1. Reduce `batch_size` in the config YAML.
2. Use gradient accumulation (not yet implemented in brainrisk, but straightforward to add).
3. Use mixed-precision training (`torch.cuda.amp`) — reduces memory by ~30–40%.
4. Reduce model size (`embed_dim`, `num_layers`).

For reference, the MINiT with the paper's configuration (D=256, L=6, N_H=8) fits comfortably on a V100 32 GB at batch size 8–16 with 64³ input volumes.

### Module load order matters

Some clusters have conflicting module dependencies. A safe load order:

```bash
module purge              # start clean
module load gcc/12.3.0    # compiler first
module load cuda/11.8.0   # CUDA before anything that depends on it
module load nccl/2.18.1   # NCCL after CUDA
# Then activate conda (which provides PyTorch, Python, etc.)
```

Loading modules *after* activating conda can sometimes shadow conda's libraries. If you encounter version conflicts, try `module load` before `conda activate`.

---

## 7. Recommended SLURM Workflow

A typical experiment cycle on HPC:

```
1. Edit config YAML locally (or on the cluster).
2. Submit training job:
      sbatch --export=CONFIG=configs/vit_subtype_classification.yaml \
             infrastructure/slurm_template.sh
3. Monitor:
      squeue -u $USER                    # job status
      tail -f logs/brainrisk-train_*.out # live output
4. Inspect results:
      cat artifacts/dl/training_log.jsonl | python -m json.tool
5. Resume from best checkpoint (if needed):
      # Set resume_path in the config YAML, then resubmit.
6. Copy final artifacts to persistent storage.
```

### Job arrays for hyperparameter sweeps

For systematic hyperparameter exploration, create a directory of config variants and submit an array:

```bash
# Generate config variants (e.g., varying learning rate).
for lr in 0.0001 0.00005 0.00001; do
    sed "s/learning_rate: .*/learning_rate: ${lr}/" \
        configs/vit_subtype_classification.yaml \
        > configs/sweep/lr_${lr}.yaml
done

# Submit as array (one job per config).
ls configs/sweep/*.yaml > configs/sweep/manifest.txt
#SBATCH --array=1-3
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/sweep/manifest.txt)
# ... rest of slurm_template.sh
```
