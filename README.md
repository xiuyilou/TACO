# TACO

Implementation of When Implausible Tokens Get Reinforced: Tail-Aware
Credit Calibration for LLM Reinforcement Learning
## What was changed

- `verl/trainer/ppo/core_algos.py`
  - Added `compute_taco_token_advantages(...)`.
  - The implementation matches the paper formula:
    - `r_tail = -log p - H + log(alpha)`
    - `w = 1` if `r_tail <= 0`
    - `w = 1 - lambda * (1 - exp(-r_tail))` if `r_tail > 0`
    - Only tokens with positive sequence-level advantage are reweighted.
    - Non-positive advantages are preserved unchanged.

- `verl/workers/actor/dp_actor.py`
  - Replaced the previous positive-feasible reweighting logic with standard TACO.
  - TACO is applied using current-policy `log_prob` and `entropy` before the PPO clipped surrogate update.

- `verl/workers/config/actor.py`
- `verl/trainer/config/actor/actor.yaml`
  - Replaced the previous custom knobs with:
    - `taco_enable`
    - `taco_alpha`
    - `taco_lambda`

- `run.sh`
  - Updated to use the standard TACO config names.
  - Removed hardcoded secrets and experiment identifiers.
  - Default logging is now console-only for anonymous release.

## How to run

Set the model path explicitly, then launch:

```bash
export MODEL_PATH=/path/to/model_or_actor_checkpoint
bash run.sh
```

Useful optional variables:

```bash
export TRAIN_PATH=/path/to/train_data
export VAL_PATH=/path/to/val_data
export PROJECT_NAME=taco
export RUN_NAME_BASE=grpo_taco
export TACO_ENABLE=true
export TACO_ALPHA=0.01
export TACO_LAMBDA=0.9
```

To enable WandB manually:

```bash
export ENABLE_WANDB=true
export WANDB_MODE=online
export WANDB_API_KEY=<optional_wandb_key>
bash run.sh
```

## Notes

- The provided `run.sh` uses the FSDP2 actor path, where the TACO update is implemented in `verl/workers/actor/dp_actor.py`.
- This folder intentionally keeps only the code needed for the anonymous TACO release and strips identifying experiment metadata from the launcher.
