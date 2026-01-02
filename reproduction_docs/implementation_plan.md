# Replication Plan: Shadow Hand Bounce (PPO + FD)

## Goal Description
Reproduce the "Bounce" task results from Miller et al., 2025 by aligning the `isaaclab_rl` and `roto` repositories with the paper's specific configurations. The target metric is ~79 bounces in 10 seconds.

## User Review Required
> [!IMPORTANT]
> **GPU Infrastructure Required**: All verification and training steps (including the Dry Run) **require an NVIDIA GPU** to run Isaac Lab/Isaac Sim.
> - **Action Required**: Please ensure you have access to an NVIDIA GPU node (e.g., AWS G5/P3 instance or similar) with the Isaac Lab environment configured.
> - **Workflow**: I will perform code edits here. You will need to pull these changes to your GPU node to execute the training.

> [!IMPORTANT]
> **Reward Function Override**: The existing `bounce.py` code only rewards bounces. I will rewrite `compute_rewards` to strictly follow Appendix E.3: `r = r_air + r_bounce + r_fall`.
>
> **Rollout Length**: Changing `agent.rollouts` from 16 to 32 as per Appendix G.
>
> **Fall Threshold**: Paper specifies 24cm for fall penalty, while code defaults to 20cm (`out_of_bounds`). I will update this to 24cm.

## Proposed Changes

### `roto` Repository

#### [Modify] [bounce.py](file:///Users/kav/ontologen/github/roto/roto/tasks/shadow/bounce.py)
- Update `BounceCfg`:
    - Change `out_of_bounds` from `0.2` to `0.24`.
- Update `BounceEnv`:
    - Ensure `timeout` matches paper (600 timesteps).
- Update `compute_rewards`:
    - Implement `r_air`: Reward proportional to `time_without_contact` (from `BounceEnv`).
    - Implement `r_fall`: Penalty if object distance > 0.24m.
    - Implement `r_bounce`: Existing logic (bonus on valid bounce).

#### [Modify] [forward_dynamics.yaml](file:///Users/kav/ontologen/github/roto/roto/tasks/shadow/agents/bounce/forward_dynamics.yaml)
- Change `agent.rollouts` to `32`.
- Verify/Update `learning_rate` and other PPO parameters if explicit values from paper are confirmed (defaulting to current file values where paper is ambiguous, but prioritizing paper-stated defaults).

### `isaaclab_rl` Repository
- No changes expected unless bug fixes are needed.

## Verification Plan

### Automated Tests
- **Dry Run (Requires GPU)**: Execute a short training run (e.g., 100 steps) to verify:
    - No syntax errors in reward function.
    - Reward components (`r_air`, `r_bounce`) are being generated (non-zero).
    - Rollout length is accepted.

### Manual Verification
- **Launch Training (Requires GPU)**: Start the full training run using the standard command (to be confirmed, likely `python scripts/rsl_rl/train.py` or similar).
