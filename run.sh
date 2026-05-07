#!/usr/bin/env bash
set -xeuo pipefail

export RAY_TMPDIR="${RAY_TMPDIR:-$HOME/TACO/ray_tmp}"
mkdir -p "$RAY_TMPDIR"

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export PYTORCH_SEED="${PYTORCH_SEED:-42}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

WORKDIR="${WORKDIR:-$HOME/TACO}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data}"

TRAIN_PATH="${TRAIN_PATH:-$DATA_DIR/dapo-17k/}"
VAL_PATH="${VAL_PATH:-$DATA_DIR/aime24/}"
MODEL_PATH="${MODEL_PATH:-}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "Please set MODEL_PATH to the actor checkpoint or HF model path."
    exit 1
fi

PROJECT_NAME="${PROJECT_NAME:-taco}"
RUN_NAME_BASE="${RUN_NAME_BASE:-grpo_taco}"
CURRENT_DATETIME="$(date +"%Y%m%d_%H%M%S")"
RUN_NAME="${RUN_NAME:-${RUN_NAME_BASE}_${CURRENT_DATETIME}}"

LOG="${LOG:-$WORKDIR/${RUN_NAME}.log}"
SAVE_CONTENTS="${SAVE_CONTENTS:-['hf_model']}"

ENABLE_WANDB="${ENABLE_WANDB:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$WORKDIR/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WORKDIR/wandb_cache}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

TRAINER_LOGGER="['console']"
if [[ "$ENABLE_WANDB" == "true" ]]; then
    TRAINER_LOGGER="['console','wandb']"
fi

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
python3 -c "import verl; print(verl.__file__)"

PARALLEL_SIZE="${PARALLEL_SIZE:-4}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
VAL_MAX_RESPONSE_LENGTH="${VAL_MAX_RESPONSE_LENGTH:-16384}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-6312}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_N="${VAL_N:-32}"

TACO_ENABLE="${TACO_ENABLE:-true}"
TACO_ALPHA="${TACO_ALPHA:-0.01}"
TACO_LAMBDA="${TACO_LAMBDA:-0.9}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="['$TRAIN_PATH']" \
    data.val_files="['$VAL_PATH']" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.val_max_response_length="${VAL_MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.taco_enable="${TACO_ENABLE}" \
    actor_rollout_ref.actor.taco_alpha="${TACO_ALPHA}" \
    actor_rollout_ref.actor.taco_lambda="${TACO_LAMBDA}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${PARALLEL_SIZE}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="${SAVE_CONTENTS}" \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKEN_LEN_PER_GPU}" \
    ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.response_length="${VAL_MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.max_resp_len=8192 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir="$WORKDIR/rollout_data/$PROJECT_NAME/$RUN_NAME" \
    "trainer.logger=${TRAINER_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=12 \
    trainer.total_training_steps=300 \
    trainer.resume_mode=auto \
    trainer.save_best_only=False \
    trainer.validation_data_dir="$WORKDIR/checkpoints/$PROJECT_NAME/$RUN_NAME/validation" \
    "$@" 2>&1 | tee "${LOG}"
