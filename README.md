<div align="center">

# Successful Trajectories, Overcredited Tokens: Calibrating Positive Updates in Reinforcement Learning with Verifiable Rewards


## 💡 About

**TACO (Tail-Aware Credit Calibration)** is a lightweight credit-calibration method for LLM reinforcement learning.

GRPO-style methods usually broadcast a sequence-level advantage to every token in a rewarded completion. This can unintentionally reinforce locally implausible low-probability tokens when they appear inside otherwise successful trajectories. TACO mitigates this issue by estimating token-level tail risk from generation-time statistics and softly reducing positive credit for high-risk tokens.


## 🛠️ Implementation

The main implementation modifies the GRPO/PPO training pipeline in `verl`.

### `verl/trainer/ppo/core_algos.py`

The implementation follows the paper formula:

```python
r_tail = -log_p - entropy + log(alpha)

if r_tail <= 0:
    w = 1
else:
    w = 1 - lambda_ * (1 - exp(-r_tail))
```

TACO only reweights tokens with positive sequence-level advantage:

```python
if advantage > 0:
    calibrated_advantage = w * advantage
else:
    calibrated_advantage = advantage
```

### `verl/workers/actor/dp_actor.py`


### Config files

TACO-related options are exposed through:

```text
verl/workers/config/actor.py
verl/trainer/config/actor/actor.yaml
```

## 🧪 Training with TACO

TACO can be enabled through environment variables before launching training:

```bash
export TACO_ENABLE=true
export TACO_ALPHA=0.01
export TACO_LAMBDA=0.9

bash run.sh
```

### TACO-specific parameters

| Parameter     | Description                                                                                              | Default |
| ------------- | -------------------------------------------------------------------------------------------------------- | ------- |
| `TACO_ENABLE` | Whether to enable TACO credit calibration                                                                | `true`  |
| `TACO_ALPHA`  | Tail-risk strictness. Smaller values are more conservative; larger values identify more tokens as risky. | `0.01`  |
| `TACO_LAMBDA` | Maximum suppression strength for high-risk positive credit.                                              | `0.9`   |
