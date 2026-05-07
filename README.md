# TACO

Implementation of When Implausible Tokens Get Reinforced: Tail-Aware
Credit Calibration for LLM Reinforcement Learning
## What was changed

- `verl/trainer/ppo/core_algos.py`
  - The implementation matches the paper formula:
    - `r_tail = -log p - H + log(alpha)`
    - `w = 1` if `r_tail <= 0`
    - `w = 1 - lambda * (1 - exp(-r_tail))` if `r_tail > 0`
    - Only tokens with positive sequence-level advantage are reweighted.
    - Non-positive advantages are preserved unchanged.

- `verl/workers/actor/dp_actor.py`
  - TACO is applied using current-policy `log_prob` and `entropy` before the PPO clipped surrogate update.

- `verl/workers/config/actor.py`
- `verl/trainer/config/actor/actor.yaml`

- `run.sh`

## TACO specific parameter

```bash
export TACO_ENABLE=true
export TACO_ALPHA=0.01
export TACO_LAMBDA=0.9
```

