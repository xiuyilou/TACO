<div align="center">
<h1 align="center">
  Tail-Aware Credit Calibration (TACO)
</h1>
<h3><em>Right Answer, Wrong Tokens: Tail-Aware Credit Calibration for RLVR</em></h3>
<p align="center" style="margin-bottom: 0px;">
    <a href="https://arxiv.org/abs/2607.07976"><img src="https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white" height="24"></a><a href="https://github.com/xiuyilou/TACO"><img src="https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white" height="24"></a>
</p>
<p align="center" style="font-size: 10px;">
    <a href="#-about" style="text-decoration: none; font-weight: bold;">💡 About</a> •
    <a href="#-clone-and-use-taco" style="text-decoration: none; font-weight: bold;">🚀 Clone and Use TACO</a> •
    <a href="#-training-with-taco" style="text-decoration: none; font-weight: bold;">🧪 Training with TACO</a> •
    <a href="#-how-taco-works" style="text-decoration: none; font-weight: bold;">🔍 How TACO Works</a> •
    <a href="#-citation" style="text-decoration: none; font-weight: bold;">💬 Citation</a>
</p>
<!-- TODO: add an overview figure, e.g. <img src="./assets/overview.png" alt="TACO Overview" width="60%"> -->
</div>

## 💡 About

**Tail-Aware Credit Calibration (TACO)** is a **lightweight** credit-calibration method for LLM reinforcement learning with verifiable rewards (RLVR). GRPO-style methods broadcast a single sequence-level advantage to every token in a rewarded completion, which can reinforce locally erroneous tokens alongside useful ones. By calibrating the positive credit assigned to such tokens, TACO delivers:

- **Consistent Performance Gains:** Improves over GRPO-style baselines across four LLMs and eight reasoning benchmarks.
- **Improved Training Stability:** Supports sustained performance gains in long-horizon RL.
- **No Extra Supervision:** Requires no token-level error labels; the policy serves as its own reference.

## 🚀 Clone and Use TACO
TACO is a **plug-in** modification of the GRPO/PPO training pipeline in [`verl`](https://github.com/volcengine/verl).

#### 1\. Environment Setup

```bash
git clone https://github.com/xiuyilou/TACO.git ~/TACO
cd ~/TACO
```

The repository ships a modified copy of `verl`, which `run.sh` adds to `PYTHONPATH`. Install the dependencies of `verl` in your environment first. `run.sh` expects the repository at `$HOME/TACO`; set `WORKDIR` to use a different location.

#### 2\. Run Example

TACO is enabled through environment variables before launching training:

```bash
export TACO_ENABLE=true
export TACO_ALPHA=0.01
export TACO_LAMBDA=0.9

bash run.sh
```

## 🧪 Training with TACO

#### 1\. TACO-specific Parameters

| Parameter     | Description                                                                                              | Default |
| :------------ | :------------------------------------------------------------------------------------------------------- | :-----: |
| `TACO_ENABLE` | Whether to enable TACO credit calibration.                                                               | `true`  |
| `TACO_ALPHA`  | Tail-risk strictness. Smaller values are more conservative; larger values identify more tokens as risky. | `0.01`  |
| `TACO_LAMBDA` | Maximum suppression strength for high-risk positive credit.                                              | `0.9`   |

#### 2\. Configuration Files

TACO-related options are exposed through:

```text
verl/workers/config/actor.py
verl/trainer/config/actor/actor.yaml
```

#### 3\. Code Structure

The main implementation modifies the GRPO/PPO training pipeline in `verl`:

* `verl/trainer/ppo/core_algos.py`: computes the tail-risk score and the calibrated advantage.
* `verl/workers/actor/dp_actor.py`: reads `taco_enable`, `taco_alpha`, and `taco_lambda` from the actor config, validates them, and applies the calibration to the advantages during the policy update before the policy loss is computed. It also enables the per-token entropy computation that TACO requires and logs the calibration statistics under `taco/*`.

## 🔍 How TACO Works

TACO estimates token-level tail risk from generation-time statistics and softly reduces positive credit for high-risk tokens. The design of TACO consists of two steps:

- **Tail-Risk Scoring**: Each sampled token's rarity is assessed relative to the policy's own local expectations, and tokens above the risk threshold receive a soft weight below one.

```python
r_tail = -log_p - entropy + log(alpha)

if r_tail <= 0:
    w = 1
else:
    w = 1 - lambda_ * (1 - exp(-r_tail))
```

- **Positive-Credit Calibration**: Only tokens with positive sequence-level advantage are reweighted; negative advantages are left unchanged.

```python
if advantage > 0:
    calibrated_advantage = w * advantage
else:
    calibrated_advantage = advantage
```

## 🫡 Acknowledgement

This codebase is built on [verl](https://github.com/volcengine/verl), an open-source RL training library for LLMs.

## 💬 Citation

If you find TACO helpful, please cite the paper and star this repo, thanks!

```bibtex
@article{lou2026taco,
  title={Right Answer, Wrong Tokens: Tail-Aware Credit Calibration for RLVR},
  author={Lou, Xiuyi and Xu, Zicheng and Chuang, Yu-Neng and Le, Hoang Anh Duy and Xu, Zhaozhuo and Wang, Guanchu and Braverman, Vladimir},
  journal={arXiv preprint arXiv:2607.07976},
  year={2026}
}
```
