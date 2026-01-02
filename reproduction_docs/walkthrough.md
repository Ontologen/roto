# Replication Walkthrough: Shadow Hand Bounce

## Changes Implemented
I have modified the `roto` repository to strictly follow the "Enhancing Tactile-based RL" paper (Appendix E & G) instead of the repository defaults.

### 1. Reward Function (`bounce.py`)
**Goal**: Match Appendix E.3: $R = r_{air} + r_{bounce} + r_{fall}$
- **New Logic**:
  - `air_reward`: Proportional to time since last contact (Scale: 0.1).
  - `bounce_reward`: Bonus for valid bounce (Scale: 10.0, existing).
  - `fall_reward`: Penalty if object distance > 24cm from target (Scale: -5.0).
- **Configuration**: Added `out_of_bounds = 0.24` (was 0.20) and reward scalars to `BounceCfg`.

### 2. Hyperparameters (`forward_dynamics.yaml`)
**Goal**: Match Appendix G (Table A4).
- **Rollout Length**: Increased from `16` to `32`.
- **Observation Stack**: Verified as `4` (matches paper).

## Execution Guide

> [!IMPORTANT]
> **GPU Required**: The following commands must be run on an NVIDIA GPU node with Isaac Lab installed.

### 1. Setup
Sync code to your GPU node and install the packages:
```bash
# In the directory where you cloned/synced the repos
cd isaaclab_rl
pip install -e .
cd ../roto
pip install -e .
```

### 2. Verification (Dry Run)
Run a short training session to verify the new reward function doesn't crash:
```bash
# Run for just a few steps to check for errors
python scripts/train.py --task Bounce --num_envs 64 --max_iterations 5 --headless --seed 42 --agent_cfg forward_dynamics
```
*Expected Output*: Training starts, logs show non-zero values for `air_reward`, `bounce_reward`, etc.

### 3. Launch Full Reproduction
Execute the full training run:
```bash
python scripts/train.py --task Bounce --num_envs 4096 --headless --seed 42 --agent_cfg forward_dynamics
```
*Expected Duration*: ~5-10 hours depending on GPU.
*Success Metric*: Look for `num_bounces` approaching **79** (or ~800 reward) after ~200M steps.

### 4. Visual Verification
You do **not** need a streaming client to verify behavior visually. You can record videos in headless mode:

```bash
# After training, use the best checkpoint
python scripts/play.py --task Bounce --num_envs 1 --agent_cfg forward_dynamics \
    --checkpoint runs/bounce/forward_dynamics/nn/bounce_forward_dynamics.pt \
    --headless --video --video_length 1000
```
*Result*: A video file (e.g., `bounce.mp4`) will be saved in the `videos/` directory. You can SCP this file to your laptop to view it.
