#!/bin/bash
set -e
echo "Starting setup..."

# Initialize conda
if ! command -v conda &> /dev/null; then
    if [ -f "/opt/conda/bin/conda" ]; then
        eval "$(/opt/conda/bin/conda shell.bash hook)"
    else
        echo "Conda not found. Installing Miniforge..."
        wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O Miniforge3.sh
        bash Miniforge3.sh -b -p $HOME/miniforge3
        eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
        rm Miniforge3.sh
    fi
fi

# Create env if not exists
if ! conda info --envs | grep -q isaaclab; then
    echo "Creating conda env 'isaaclab'..."
    conda create -n isaaclab python=3.10 -y
else
    echo "Conda env 'isaaclab' already exists."
fi
conda activate isaaclab
pip cache purge

echo "Installing system dependencies..."
sudo apt-get update
# Install graphics and utility libraries required for Isaac Lab
# Note: libnvidia-gl version must match your driver. We attempt to find the driver version or fallback to 570.
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1 | head -n 1)
sudo apt-get install -y libgl1 libglx0 libegl1 libxext6 libx11-6 libsm6 \
                        libxrender1 libxtst6 libxi6 libvulkan1 \
                        mesa-vulkan-drivers vulkan-tools \
                        libnvidia-gl-${DRIVER_VERSION}

echo "Configuring environment variables..."
# Export LD_LIBRARY_PATH to ~/.bashrc for persistence
ISAAC_LIB_PATH="\$(find \$HOME/miniforge3/envs/isaaclab/lib/python3.10/site-packages/isaacsim/extscache -type d \( -path '*/omni.usd.core*/bin' -o -path '*/omni.usd.libs*/bin' \) | tr '\n' ':')"
if ! grep -q "omni.usd.libs" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=${ISAAC_LIB_PATH}:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export WANDB_MODE=disabled" >> ~/.bashrc
fi
# Export for current session
export LD_LIBRARY_PATH=$(eval echo "${ISAAC_LIB_PATH}"):$LD_LIBRARY_PATH
export WANDB_MODE=disabled

echo "Installing Torch..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "Installing Isaac Sim binaries (Pinned to 4.2.0.2)..."
pip install isaacsim-rl==4.2.0.2 isaacsim-replicator==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 --extra-index-url https://pypi.nvidia.com

echo "Setting up workspace..."
mkdir -p ~/workspace
cd ~/workspace

# Isaac Lab
if [ ! -d "IsaacLab" ]; then
    echo "Cloning IsaacLab..."
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    echo "Patching IsaacLab setup.py for rsl-rl..."
    # NVIDIA's setup.py requests 'rsl-rl', which assumes a local clone or internal registry.
    # On PyPI, the package is named 'rsl-rl-lib'. We patch this to pull the correct package
    # from PyPI without needing to manually clone the rsl-rl repository.
    find . -name "setup.py" -print0 | xargs -0 sed -i 's/"rsl-rl"/"rsl-rl-lib==2.3.3"/g'
    echo "Installing IsaacLab dependencies..."
    ./isaaclab.sh --install
    cd ..
else
    echo "IsaacLab already cloned."
fi

# Roto and IsaacLab RL
if [ ! -d "isaaclab_rl" ]; then
    echo "Cloning isaaclab_rl..."
    git clone https://github.com/Ontologen/isaaclab_rl
else
    echo "isaaclab_rl already cloned."
fi

if [ ! -d "roto" ]; then
    echo "Cloning roto..."
    git clone https://github.com/Ontologen/roto
else
    echo "roto already cloned."
fi

# Install editable
echo "Installing editable packages..."
cd isaaclab_rl && pip install -e .
cd ../roto && pip install -e .

echo "Setup complete!"
