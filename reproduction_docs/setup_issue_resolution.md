# Setup Issue Resolution Log

This document catalogs the technical issues encountered during the setup and replication of the **Shadow Hand Bounce** task and the solutions applied to resolve them.

## 1. Infrastructure & Operating System

### Issue: `glibc` Version Mismatch
*   **Problem**: The initial attempt using a standard Deep Learning AMI failed because the pre-compiled Isaac Lab binaries required a newer version of `glibc` than what was available on the default OS.
*   **Resolution**: Switched the VM operating system to **Ubuntu 22.04 LTS**. This version provides the requisite system libraries compatible with Isaac Lab v1.2.0.

### Issue: Headless Rendering & Vulkan Support
*   **Problem**: The simulation crashed immediately in headless mode with `ERROR_INCOMPATIBLE_DRIVER` or failures to initialize the windowing system. `vulkaninfo` reported missing libraries.
*   **Resolution**: Installed a comprehensive set of graphics and utility libraries:
    ```bash
    sudo apt-get install -y libgl1 libglx0 libegl1 libxext6 libx11-6 libsm6 \
                            libxrender1 libxtst6 libxi6 libvulkan1 \
                            mesa-vulkan-drivers vulkan-tools \
                            libnvidia-gl-570  # Matched to installed driver version
    ```

## 2. Dependency Management

### Issue: `rsl-rl` Package Conflict
*   **Problem**: The `isaaclab_rl` setup script tried to install `rsl-rl`, but the available PyPI package had version conflicts with PyTorch/CUDA.
*   **Resolution**: 
    1.  Patched `omni.isaac.lab_tasks` `setup.py` to depend on `rsl-rl-lib` instead.
    2.  Pinned `rsl-rl-lib==2.3.3` to ensure compatibility with the installed PyTorch 2.5+/CUDA 12 environment.

### Issue: Missing Shared Libraries (`libomni.usd.so`)
*   **Problem**: Python could not locate core USD libraries bundled with Isaac Sim, causing `ImportError`.
*   **Resolution**: Manually exported the `LD_LIBRARY_PATH` to include the extension cache directories:
    ```bash
    export LD_LIBRARY_PATH=$(find ~/miniforge3/envs/isaaclab/lib/python3.10/site-packages/isaacsim/extscache -type d \( -path '*/omni.usd.core*/bin' -o -path '*/omni.usd.libs*/bin' \) | tr '\n' ':')\$LD_LIBRARY_PATH
    ```

## 3. Codebase & API Compatibility

### Issue: Legacy Imports
*   **Problem**: The `roto` codebase used legacy import paths (e.g., `from isaaclab...`) that were removed in Isaac Lab v1.0+.
*   **Resolution**: Refactored imports across `bounce.py`, `shadow.py`, `roto_env.py`, and `shadow_hand.py` to use the new namespace `omni.isaac.lab`.

### Issue: Broken Asset Path (Shadow Hand)
*   **Problem**: The simulation failed with "USD file not found". `roto` was referencing `Robots/ShadowRobot/ShadowHand`, but the Isaac Lab 4.2 asset structure is `Robots/ShadowHand`.
*   **Resolution**: Corrected the `usd_path` in `roto/assets/shadow_hand.py` to remove the redundant `ShadowRobot` directory.

### Issue: Deprecated Joint API
*   **Problem**: `AttributeError: ... has no attribute 'joint_vel_limits'`. The API for accessing articulation data changed in newer Isaac Lab versions.
*   **Resolution**: Updated `roto_env.py` to use `soft_joint_vel_limits` instead of deprecated `joint_vel_limits`.

## 4. Runtime & Configuration Errors

### Issue: Memory Allocation Crash (`num_envs`)
*   **Problem**: `ValueError: need at least one array to concatenate`.
*   **Cause**: The default configuration set `num_eval_envs = 100`. Running with `--num_envs 64` resulted in `num_training_envs = 64 - 100 = -36` (effectively 0), causing empty memory buffer initialization.
*   **Resolution**:
    -   **Dry Run**: Executed with `--num_envs 128` (giving 28 training envs) to satisfy the evaluation requirement.
    -   **Full Run**: Should use `4096` envs as per paper specs.

### Issue: Missing Config Attribute (`act_moving_average`)
*   **Problem**: `AttributeError: 'BounceCfg' object has no attribute 'act_moving_average'`.
*   **Resolution**: Added `act_moving_average = 1.0` to the `BounceCfg` class in `bounce.py` (value 1.0 disables smoothing, preventing side effects during reproduction).

### Issue: Reward Logging KeyError
*   **Problem**: `KeyError: 'fall_reward'` during the logging phase of the training loop.
*   **Resolution**: Commented out the specific line in `bounce.py` logging `fall_reward` to `extras["log"]`. The reward logic itself remains active in `total_reward`, so training is unaffected.

### Issue: WandB Blocking Execution
*   **Problem**: The script hung waiting for an API key input.
*   **Resolution**: Set `export WANDB_MODE=disabled` in the execution environment to bypass the login requirement for verification runs.
