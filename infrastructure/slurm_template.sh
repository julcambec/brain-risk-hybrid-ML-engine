#!/bin/bash
# -------------------------------------------------------------
# brainrisk — SLURM job template for MINiT DDP training
# -------------------------------------------------------------
#
# This template demonstrates the HPC job configuration used during
# the original MINiT experiments on UBC Sockeye (NVIDIA V100 32 GB).
# Adapt resource requests, paths, and module names to your cluster.
#
# Usage:
#   sbatch slurm_template.sh                          # defaults
#   sbatch --export=CONFIG=configs/vit_sex_classification.yaml slurm_template.sh
#   sbatch --export=CONFIG=configs/vit_subtype_classification.yaml,GPUS=2 slurm_template.sh
#
# The brainrisk training loop (src/brainrisk/deeplearning/trainer.py)
# detects DDP via the RANK and WORLD_SIZE environment variables that
# torchrun sets automatically.  It selects NCCL when CUDA is available
# and falls back to Gloo on CPU-only nodes.
# -------------------------------------------------------------

# --- Resource requests ---
#SBATCH --job-name=brainrisk-train
#SBATCH --account=<allocation>            # your cluster allocation
#SBATCH --nodes=1                         # single-node DDP (multi-node below)
#SBATCH --ntasks-per-node=1               # torchrun handles process spawning
#SBATCH --cpus-per-task=8                 # data-loader workers + overhead
#SBATCH --gpus-per-node=4                 # V100 × 4 → effective batch = 4 × per-GPU batch
#SBATCH --constraint=gpu_mem_32           # request 32 GB GPU memory (cluster-specific)
#SBATCH --mem=32G                         # host RAM (covers data loading + preprocessing)
#SBATCH --time=8:00:00                    # wall-clock limit (adjust per experiment)
#SBATCH --output=logs/%x_%j.out           # stdout → logs/<job-name>_<job-id>.out
#SBATCH --error=logs/%x_%j.err            # stderr → logs/<job-name>_<job-id>.err
#SBATCH --mail-user=<email>               # your email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL

# --- User-configurable variables (override with --export) ---
CONFIG="${CONFIG:-configs/vit_subtype_classification.yaml}"
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
RESUME="${RESUME:-}"                      # path to checkpoint for resuming

# --- Stagger array starts (prevents filesystem contention) ---
sleep $(( RANDOM % 10 ))

# --- Environment setup ---
# Load cluster modules; names are site-specific; adapt for your HPC.
# The key requirement is a CUDA toolkit version compatible with your
# PyTorch build (see infrastructure/hpc_setup.md for version matrix).
module purge
module load gcc/12.3.0                    # compiler toolchain
module load cuda/11.8.0                   # must match pytorch-cuda channel
module load nccl/2.18.1                   # NCCL for multi-GPU communication

# Activate the conda environment.
# On Sockeye the base conda lives under the user's home directory.
# Adjust the path to wherever your Miniforge / Miniconda is installed.
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate brainrisk

# Ensure the conda environment's binaries take precedence.
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# --- Performance tuning ---
# OMP_NUM_THREADS: Prevent NumPy / MKL from over-subscribing CPUs.
# One thread per dataloader worker is a safe default; PyTorch uses
# its own thread pool for compute kernels.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# NCCL tuning (optional, may be cluster-dependent):
# export NCCL_IB_DISABLE=1              # disable InfiniBand if unsupported
# export NCCL_SOCKET_IFNAME=eth0        # force a specific network interface
# export NCCL_DEBUG=INFO                # verbose NCCL logging for debugging

# --- Diagnostic banner ---
echo "================================="
echo "  brainrisk — DDP Training Job"
echo "================================="
echo "  Job ID        : ${SLURM_JOB_ID}"
echo "  Node(s)       : ${SLURM_NODELIST}"
echo "  GPUs/node     : ${GPUS}"
echo "  CPUs/task     : ${SLURM_CPUS_PER_TASK}"
echo "  Config        : ${CONFIG}"
echo "  Resume from   : ${RESUME:-<none>}"
echo "  Python        : $(which python)"
echo "  PyTorch       : $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA (torch)  : $(python -c 'import torch; print(torch.version.cuda)')"
echo "  CUDA visible  : ${CUDA_VISIBLE_DEVICES:-all}"
echo "  NCCL version  : $(python -c 'import torch; print(torch.cuda.nccl.version())' 2>/dev/null || echo 'N/A')"
echo ""=================================""

# Verify GPU visibility before launching.
python -c "
import torch
n = torch.cuda.device_count()
print(f'  Visible CUDA devices: {n}')
for i in range(n):
    print(f'    [{i}] {torch.cuda.get_device_name(i)}  '
          f'({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
"

# --- Create output directories ---
mkdir -p logs artifacts/dl/checkpoints artifacts/dl/logs

# --- Launch training ---
# torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, and MASTER_ADDR/PORT
# automatically. The brainrisk trainer (deeplearning/utils.py)
# reads these variables in setup_distributed() and initialises the
# process group with the NCCL backend when CUDA is available.
#
# Key torchrun flags:
#   --standalone        single-node mode
#   --nproc_per_node    one process per GPU
#   --nnodes            total nodes (use >1 for multi-node; see below)

TRAIN_CMD="python -m brainrisk demo-dl --config ${CONFIG}"

# Append --resume-path if a checkpoint was specified.
if [ -n "${RESUME}" ]; then
    echo "  Resuming from checkpoint: ${RESUME}"
    # Note: the brainrisk CLI does not currently accept --resume-path
    # as a CLI flag.  To resume, set 'resume_path' in the YAML config
    # or pass it via an environment variable that the trainer reads.
    # This block is included as a template for future CLI extension.
    export BRAINRISK_RESUME_PATH="${RESUME}"
fi

torchrun \
    --standalone \
    --nproc_per_node="${GPUS}" \
    -m brainrisk demo-dl \
    --config "${CONFIG}"

EXIT_CODE=$?

# --- Post-training ---
echo ""
echo "===================================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Training completed successfully."
else
    echo "  Training exited with code ${EXIT_CODE}."
fi
echo "  Checkpoints : artifacts/dl/checkpoints/"
echo "  Training log: artifacts/dl/training_log.jsonl"
echo "===================================================="

exit ${EXIT_CODE}


# -------------------------------------------------------------
# MULTI-NODE VARIANT
# -------------------------------------------------------------
# For training across multiple nodes (e.g., 2 nodes × 4 GPUs = 8 GPUs),
# replace the resource block and torchrun invocation:
#
#   #SBATCH --nodes=2
#   #SBATCH --ntasks-per-node=1
#   #SBATCH --gpus-per-node=4
#
#   # SLURM sets SLURM_NODELIST; extract the first node as master.
#   MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)
#   MASTER_PORT=29500   # or just some available port within the valid user port range
#
#   srun torchrun \
#       --nnodes=${SLURM_NNODES} \
#       --nproc_per_node=${GPUS} \
#       --rdzv_id=${SLURM_JOB_ID} \
#       --rdzv_backend=c10d \
#       --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
#       -m brainrisk demo-dl \
#       --config "${CONFIG}"
#
# The brainrisk DDP utilities (deeplearning/utils.py) handle this
# transparently: torchrun sets RANK, WORLD_SIZE, and LOCAL_RANK,
# and setup_distributed() initialises the process group accordingly.
# -------------------------------------------------------------
