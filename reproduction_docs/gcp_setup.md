# GCP L4 Setup Guide for Isaac Lab

This guide helps you provision a **GCP L4 GPU** instance to run the "Bounce" replication task.

## 1. Instance Selection
Target metrics: *16GB VRAM, 32GB RAM, 8 CPU cores*.

**Recommended Instance**: `g2-standard-8`
- **GPU**: 1x NVIDIA L4 (24GB VRAM) - Excellent for rendering and ML.
- **vCPUs**: 8
- **RAM**: 32 GB
- **Cost**: ~$0.70 - $1.00/hr (depending on region/spot availability).

## 1.1 Prerequisites: GPU Quotas
New GCP projects often have a default GPU quota of 0. You will likely need to request an increase before you can launch an instance.

**Required Quotas**:
1.  **GPUs (all regions)**: Global limit. Request **1**.
    *   *Metric*: `compute.googleapis.com/gpus_all_regions`
2.  **NVIDIA L4 GPUs**: Per-region limit (e.g., `us-west4`). Request **1** if not automatically provided after getting all-region quota approved.
    *   *Metric*: `compute.googleapis.com/nvidia_l4_gpus`

**How to Request**:
1.  Go to **IAM & Admin** > **Quotas** in the GCP Console.
2.  Filter for "GPU".
3.  Select the checkboxes for the metrics above.
4.  Click **Edit Quotas**, enter the new limit (1), and submit.
    *   *Note*: Approval is usually automatic or takes a few hours. If stuck, try requesting a T4 GPU (`nvidia_t4_gpus`) instead, or move to AWS.

## 2. Launch Instance (Console Steps)

1.  **Go to Compute Engine** -> **VM Instances** -> **Create Instance**.
2.  **Machine Configuration**:
    *   **Series**: `G2`
    *   **Machine Type**: `g2-standard-8` (1 L4 GPU, 8 vCPUs, 32GB Memory).
    *   *Note*: **Check availability first**. L4 GPUs are not available in all regions (e.g., `us-west-2` often lacks them, while `us-west4` is a known good region as of 2026-01-02). Always check the latest [GCP GPU regions](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones) list.
3.  **Boot Disk**:
    *   **OS**: **Deep Learning VM with CUDA 12.1** on **Ubuntu 22.04** (Important: Debian 11 is incompatible).
    *   **Type**: Balanced Persistent Disk or SSD.
    *   **Size**: At least **200 GB**.
4.  **Firewall**:
    *   Allow standard SSH (usually on by default).

## 3. Initial Setup

### Connect
Use standard SSH or gcloud:
```bash
gcloud compute ssh --zone <ZONE> <INSTANCE_NAME>
```

### Install Isaac Lab
The Deep Learning VM comes with Drivers and Conda pre-installed.

1.  **Verify GPU**:
    ```bash
    nvidia-smi  # Should show the L4 GPU
    ```

2.  **Set up Environment**:
    ```bash
    # Create clean env
    conda create -n isaaclab python=3.10 -y
    conda activate isaaclab
    
    # Install Torch compatible with Isaac Lab (Isaac Lab 4.2+ usually likes Torch 2.4.0+ cuda 11.8 or 12.1)
    # Check https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    
    # Install Isaac Sim backend
    pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit --extra-index-url https://pypi.nvidia.com
    ```

3.  **Clone Repositories**:
    ```bash
    mkdir workspace && cd workspace
    
    # Clone Isaac Lab framework
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    ./isaaclab.sh --install
    
    # Clone Project Repos
    cd ..
    git clone https://github.com/elle-miller/isaaclab_rl
    git clone https://github.com/elle-miller/roto
    ```

## 4. Syncing the Code
Transfer the modified code to your instance.

**Option A: Gcloud SCP (Easiest)**
1.  On local machine:
    ```bash
    git diff > reproduction.patch
    gcloud compute scp reproduction.patch <INSTANCE_NAME>:~/workspace/ --zone <ZONE>
    ```
2.  On VM:
    ```bash
    cd ~/workspace/roto
    git apply ~/workspace/reproduction.patch
    ```

**Option B: Manual Edit**
Update `roto/tasks/shadow/bounce.py` and `forward_dynamics.yaml` using `vim`/`nano` on the VM as described in the Walkthrough.

## 5. Running
Proceed with the **Execution Guide** in the Walkthrough.
```bash
cd ~/workspace/isaaclab_rl && pip install -e .
cd ~/workspace/roto && pip install -e .
```
