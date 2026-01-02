#!/bin/bash
set -e
echo "Starting setup..."

# Initialize conda (Deep Learning VM location)
eval "$(/opt/conda/bin/conda shell.bash hook)"

# Create env if not exists
if ! conda info --envs | grep -q isaaclab; then
    echo "Creating conda env 'isaaclab'..."
    conda create -n isaaclab python=3.10 -y
else
    echo "Conda env 'isaaclab' already exists."
fi
conda activate isaaclab

echo "Installing Torch..."
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "Installing Isaac Sim binaries..."
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit --extra-index-url https://pypi.nvidia.com

echo "Setting up workspace..."
mkdir -p ~/workspace
cd ~/workspace

# Isaac Lab
if [ ! -d "IsaacLab" ]; then
    echo "Cloning IsaacLab..."
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
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
