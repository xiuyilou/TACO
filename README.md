<div align="center">

# When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning

<p>
  <a href="YOUR_ARXIV_LINK"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="YOUR_GITHUB_LINK"><img src="https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
</p>

<em>Calibrate token-level credit in LLM reinforcement learning by suppressing unreliable positive updates from implausible tail tokens.</em>

<p>
  <a href="#-about">About</a> ·
  <a href="#-implementation">Implementation</a> ·
  <a href="#-training-with-taco">Training with TACO</a> ·
</p>

</div>

## 🎉 Updates

- **[07/05/2026]** Initial open-source release of TACO built on top of [verl](https://github.com/volcengine/verl).

## 💡 About

**TACO (Tail-Aware Credit Calibration)** is a lightweight credit-calibration method for LLM reinforcement learning.

GRPO-style methods usually broadcast a sequence-level advantage to every token in a rewarded completion. This can unintentionally reinforce locally implausible low-probability tokens when they appear inside otherwise successful trajectories. TACO mitigates this issue by estimating token-level tail risk from generation-time statistics and softly reducing positive credit for high-risk tokens.

TACO uses two local signals:

- **Sampled-token probability**: low-probability tokens are more likely to come from the unreliable tail of the policy distribution.
- **Local entropy**: entropy provides a context-dependent reference for whether a low-probability token is actually surprising under the current distribution.

Based on these signals, TACO suppresses unreliable positive credit while preserving useful rare-token exploration.


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

TACO is applied using current-policy `log_prob` and `entropy` before the PPO clipped surrogate update.

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

## 🫡 Acknowledgements

This project builds on [verl](https://github.com/volcengine/verl), an open-source reinforcement learning framework for large language models.
