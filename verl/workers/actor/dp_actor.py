# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import math
import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_taco_token_advantages, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    @staticmethod
    def _compute_response_target_rows(indices, batch_size: int, seqlen: int, response_length: int):
        if response_length <= 0:
            return None
        device = indices.device
        full_positions = (
            torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1) * seqlen
            + torch.arange(seqlen - response_length - 1, seqlen - 1, device=device, dtype=torch.long).unsqueeze(0)
        ).reshape(-1)
        inverse = torch.full((batch_size * seqlen,), -1, dtype=torch.long, device=device)
        inverse[indices] = torch.arange(indices.numel(), device=device, dtype=torch.long)
        return inverse[full_positions].reshape(batch_size, response_length)

    @staticmethod
    def _collapse_padded_flat_support(flat_topk_indices_batch, support_offsets_batch):
        if flat_topk_indices_batch is None or support_offsets_batch is None:
            return flat_topk_indices_batch, support_offsets_batch
        if flat_topk_indices_batch.dim() != 2 or support_offsets_batch.dim() != 2:
            return flat_topk_indices_batch, support_offsets_batch

        device = flat_topk_indices_batch.device
        lengths = support_offsets_batch[:, -1].to(dtype=torch.int64)
        total_len = int(lengths.sum().item())
        if total_len == 0:
            collapsed_offsets = torch.zeros_like(support_offsets_batch, dtype=torch.int64, device=device)
            return torch.empty(0, dtype=flat_topk_indices_batch.dtype, device=device), collapsed_offsets

        flat_segments = []
        collapsed_offsets = torch.zeros_like(support_offsets_batch, dtype=torch.int64, device=device)
        base = 0
        for idx in range(flat_topk_indices_batch.shape[0]):
            cur_len = int(lengths[idx].item())
            cur_offsets = support_offsets_batch[idx].to(dtype=torch.int64, device=device)
            if cur_len > 0:
                flat_segments.append(flat_topk_indices_batch[idx, :cur_len])
            collapsed_offsets[idx] = cur_offsets + base
            base += cur_len

        return torch.cat(flat_segments, dim=0), collapsed_offsets

    @staticmethod
    def _make_rollout_sampling_mask_logits_fn(topk_idx_batch, keep_flags_batch=None, support_counts_batch=None):
        from verl.utils.torch_functional import logprobs_from_logits_masked

        def _scatter_flags_to_mask(out, indices, flags):
            """scatter_add_ based OR: handles duplicate indices correctly."""
            acc = torch.zeros(out.shape, dtype=torch.float32, device=out.device)
            acc.scatter_add_(-1, indices, flags.float())
            out |= acc > 0

        def fn(logits, labels, target_rows=None):
            indices_device = logits.device
            topk_idx_local = topk_idx_batch.to(device=indices_device, dtype=torch.int64)
            keep_flags_local = None
            if keep_flags_batch is not None:
                keep_flags_local = keep_flags_batch.to(device=indices_device, dtype=torch.bool)
            elif support_counts_batch is not None:
                support_counts_local = support_counts_batch.to(device=indices_device, dtype=torch.int64)
                keep_flags_local = (
                    torch.arange(topk_idx_local.shape[-1], device=indices_device)
                    < support_counts_local.unsqueeze(-1)
                )
            mask = torch.zeros_like(logits, dtype=torch.bool)
            if logits.dim() == 2 and topk_idx_local.dim() == 3:
                tk = topk_idx_local.reshape(-1, topk_idx_local.shape[-1])
                if keep_flags_local is not None:
                    fk = keep_flags_local.reshape(-1, keep_flags_local.shape[-1])
                else:
                    fk = None
                if target_rows is not None:
                    result = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
                    target_rows = target_rows.reshape(-1).to(device=logits.device, dtype=torch.long)
                    position_count = min(tk.shape[0], target_rows.shape[0])
                    target_rows = target_rows[:position_count]
                    valid_rows = (target_rows >= 0) & (target_rows < logits.shape[0])
                    if torch.any(valid_rows):
                        row_indices = target_rows[valid_rows]
                        sub_logits = logits.index_select(0, row_indices)
                        sub_labels = labels.index_select(0, row_indices)
                        sub_tk = tk[:position_count][valid_rows]
                        mask_sub = torch.zeros(
                            sub_tk.shape[0], logits.shape[-1], dtype=torch.bool, device=logits.device
                        )
                        if fk is not None:
                            sub_fk = fk[:position_count][valid_rows]
                            _scatter_flags_to_mask(mask_sub, sub_tk, sub_fk)
                        else:
                            mask_sub.scatter_(-1, sub_tk, True)
                        mask_sub.scatter_(-1, sub_labels.unsqueeze(-1), True)
                        sub_log_probs = logprobs_from_logits_masked(sub_logits, sub_labels, mask_sub)
                        result.index_copy_(0, row_indices, sub_log_probs)
                    return result
                if tk.shape[0] <= logits.shape[0]:
                    mask_sub = torch.zeros(tk.shape[0], logits.shape[-1], dtype=torch.bool, device=logits.device)
                    if fk is not None:
                        _scatter_flags_to_mask(mask_sub, tk, fk)
                    else:
                        mask_sub.scatter_(-1, tk, True)
                    labels_sub = labels[:tk.shape[0]]
                    mask_sub.scatter_(-1, labels_sub.unsqueeze(-1), True)
                    mask[:tk.shape[0]] = mask_sub
                    mask[tk.shape[0] :] = True
                else:
                    mask[:] = True
            else:
                if keep_flags_local is not None:
                    _scatter_flags_to_mask(mask, topk_idx_local, keep_flags_local)
                else:
                    mask.scatter_(-1, topk_idx_local, True)
                mask.scatter_(-1, labels.unsqueeze(-1), True)
            return logprobs_from_logits_masked(logits, labels, mask)

        fn.supports_target_rows = True
        return fn

    @staticmethod
    def _make_rollout_support_subset_logprob_fn(flat_topk_indices_batch, support_offsets_batch):
        def fn(logits, labels, target_rows=None):
            device = logits.device
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = labels.reshape(-1).to(device=device, dtype=torch.int64)
            flat_topk_local = flat_topk_indices_batch.to(device=device, dtype=torch.int64)
            support_offsets_local = support_offsets_batch.to(device=device, dtype=torch.int64)
            result_dtype = torch.float32 if flat_logits.dtype in (torch.float16, torch.bfloat16) else flat_logits.dtype

            if support_offsets_local.dim() != 2:
                raise ValueError(
                    f"support_offsets must be rank-2 [bsz, response_len + 1], got {support_offsets_local.shape}"
                )

            response_len = support_offsets_local.shape[1] - 1
            num_positions = support_offsets_local.shape[0] * max(response_len, 0)
            if target_rows is not None:
                target_rows = target_rows.reshape(-1).to(device=device, dtype=torch.long)
                process_len = min(num_positions, target_rows.shape[0])
                result = torch.zeros(flat_logits.shape[0], dtype=result_dtype, device=device)
                target_position_mask = torch.zeros(process_len, dtype=torch.bool, device=device)
                position_rows = torch.arange(process_len, device=device)
                valid_target_positions = (target_rows[:process_len] >= 0) & (target_rows[:process_len] < flat_logits.shape[0])
            else:
                process_len = min(num_positions, flat_logits.shape[0], flat_labels.shape[0])
                result = torch.empty(flat_logits.shape[0], dtype=result_dtype, device=device)
                processed_mask = torch.zeros(flat_logits.shape[0], dtype=torch.bool, device=device)

            if process_len > 0 and response_len > 0:
                start_offsets = support_offsets_local[:, :-1].reshape(-1)[:process_len]
                support_lengths = (support_offsets_local[:, 1:] - support_offsets_local[:, :-1]).reshape(-1)[:process_len]
                position_rows = torch.arange(process_len, device=device)
                batch_rows = position_rows // response_len
                if target_rows is None:
                    valid_target_positions = torch.ones(process_len, dtype=torch.bool, device=device)

                positive_rows = position_rows[(support_lengths > 0) & valid_target_positions]
                if positive_rows.numel() > 0:
                    for cur_len in torch.unique(support_lengths[positive_rows], sorted=True).tolist():
                        rows = position_rows[(support_lengths == cur_len) & valid_target_positions]
                        if rows.numel() == 0:
                            continue
                        starts = start_offsets[rows]
                        gather_offsets = starts.unsqueeze(1) + torch.arange(cur_len, device=device).unsqueeze(0)
                        if flat_topk_local.dim() == 1:
                            support_ids = flat_topk_local[gather_offsets]
                        else:
                            support_ids = flat_topk_local[batch_rows[rows]].gather(1, gather_offsets)
                        if target_rows is not None:
                            actual_rows = target_rows[rows]
                        else:
                            actual_rows = rows
                        logits_rows = flat_logits[actual_rows].to(dtype=result_dtype)
                        support_logits = logits_rows.gather(1, support_ids)
                        label_ids = flat_labels[actual_rows]
                        label_logits = logits_rows.gather(1, label_ids.unsqueeze(1)).squeeze(1)

                        # The sampled label should normally already be in the stored support.
                        label_in_support = (support_ids == label_ids.unsqueeze(1)).any(dim=1)
                        aug_logits = torch.cat((support_logits, label_logits.unsqueeze(1)), dim=1)
                        if label_in_support.any():
                            aug_logits[label_in_support, -1] = torch.finfo(aug_logits.dtype).min
                        denom = torch.logsumexp(aug_logits, dim=1)
                        result[actual_rows] = label_logits - denom
                        if target_rows is not None:
                            target_position_mask[rows] = True
                        else:
                            processed_mask[rows] = True

            if target_rows is not None:
                fallback_positions = position_rows[valid_target_positions & (~target_position_mask)]
                fallback_rows = target_rows[fallback_positions]
            else:
                fallback_rows = (~processed_mask).nonzero(as_tuple=False).flatten()
            if fallback_rows.numel() > 0:
                fallback_logits = flat_logits[fallback_rows].to(dtype=result_dtype)
                fallback_labels = flat_labels[fallback_rows]
                fallback_label_logits = fallback_logits.gather(1, fallback_labels.unsqueeze(1)).squeeze(1)
                result[fallback_rows] = fallback_label_logits - torch.logsumexp(fallback_logits, dim=-1)

            return result.reshape(labels.shape)

        fn.supports_target_rows = True
        return fn

    @staticmethod
    def _make_rollout_feasible_set_mixture_logprob_fn(
        flat_topk_indices_batch,
        support_offsets_batch,
        t_in: float,
        rho: float,
    ):
        if t_in <= 0:
            raise ValueError(f"keep_sampling_feasible_set_t_in must be > 0, got {t_in}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"keep_sampling_feasible_set_rho must be in [0, 1], got {rho}")

        log_rho = float("-inf") if rho <= 0.0 else math.log(rho)
        log_one_minus_rho = float("-inf") if rho >= 1.0 else math.log1p(-rho)

        def fn(logits, labels, target_rows=None):
            device = logits.device
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = labels.reshape(-1).to(device=device, dtype=torch.int64)
            flat_topk_local = flat_topk_indices_batch.to(device=device, dtype=torch.int64)
            support_offsets_local = support_offsets_batch.to(device=device, dtype=torch.int64)
            result_dtype = torch.float32 if flat_logits.dtype in (torch.float16, torch.bfloat16) else flat_logits.dtype

            if support_offsets_local.dim() != 2:
                raise ValueError(
                    f"support_offsets must be rank-2 [bsz, response_len + 1], got {support_offsets_local.shape}"
                )

            response_len = support_offsets_local.shape[1] - 1
            num_positions = support_offsets_local.shape[0] * max(response_len, 0)
            if target_rows is not None:
                target_rows = target_rows.reshape(-1).to(device=device, dtype=torch.long)
                process_len = min(num_positions, target_rows.shape[0])
                result = torch.zeros(flat_logits.shape[0], dtype=result_dtype, device=device)
                target_position_mask = torch.zeros(process_len, dtype=torch.bool, device=device)
                position_rows = torch.arange(process_len, device=device)
                valid_target_positions = (target_rows[:process_len] >= 0) & (target_rows[:process_len] < flat_logits.shape[0])
            else:
                process_len = min(num_positions, flat_logits.shape[0], flat_labels.shape[0])
                result = torch.empty(flat_logits.shape[0], dtype=result_dtype, device=device)
                processed_mask = torch.zeros(flat_logits.shape[0], dtype=torch.bool, device=device)

            if process_len > 0 and response_len > 0:
                start_offsets = support_offsets_local[:, :-1].reshape(-1)[:process_len]
                support_lengths = (support_offsets_local[:, 1:] - support_offsets_local[:, :-1]).reshape(-1)[:process_len]
                position_rows = torch.arange(process_len, device=device)
                batch_rows = position_rows // response_len
                if target_rows is None:
                    valid_target_positions = torch.ones(process_len, dtype=torch.bool, device=device)

                candidate_rows = position_rows[valid_target_positions]
                if candidate_rows.numel() > 0:
                    if target_rows is not None:
                        candidate_actual_rows = target_rows[candidate_rows]
                    else:
                        candidate_actual_rows = candidate_rows
                    candidate_logits = flat_logits[candidate_actual_rows].to(dtype=result_dtype)
                    candidate_labels = flat_labels[candidate_actual_rows]
                    candidate_label_logits = candidate_logits.gather(1, candidate_labels.unsqueeze(1)).squeeze(1)
                    candidate_base_logprob = candidate_label_logits - torch.logsumexp(candidate_logits, dim=-1)
                    result[candidate_actual_rows] = candidate_base_logprob + log_one_minus_rho

                positive_rows = position_rows[(support_lengths > 0) & valid_target_positions]
                if positive_rows.numel() > 0 and rho > 0.0:
                    for cur_len in torch.unique(support_lengths[positive_rows], sorted=True).tolist():
                        rows = position_rows[(support_lengths == cur_len) & valid_target_positions]
                        if rows.numel() == 0:
                            continue
                        starts = start_offsets[rows]
                        gather_offsets = starts.unsqueeze(1) + torch.arange(cur_len, device=device).unsqueeze(0)
                        if flat_topk_local.dim() == 1:
                            support_ids = flat_topk_local[gather_offsets]
                        else:
                            support_ids = flat_topk_local[batch_rows[rows]].gather(1, gather_offsets)
                        if target_rows is not None:
                            actual_rows = target_rows[rows]
                        else:
                            actual_rows = rows
                        logits_rows = flat_logits[actual_rows].to(dtype=result_dtype)
                        label_ids = flat_labels[actual_rows]
                        label_logits = logits_rows.gather(1, label_ids.unsqueeze(1)).squeeze(1)
                        base_logprob = label_logits - torch.logsumexp(logits_rows, dim=-1)
                        label_in_support = (support_ids == label_ids.unsqueeze(1)).any(dim=1)
                        if torch.any(label_in_support):
                            support_logits = logits_rows.gather(1, support_ids).to(torch.float32)
                            support_logits.div_(float(t_in))
                            support_log_denom = torch.logsumexp(support_logits, dim=1)
                            support_logprob = label_logits.to(torch.float32) / float(t_in) - support_log_denom
                            mixture_logprob = torch.logaddexp(
                                base_logprob.to(torch.float32) + log_one_minus_rho,
                                support_logprob + log_rho,
                            )
                            selected_rows = actual_rows[label_in_support]
                            result[selected_rows] = mixture_logprob[label_in_support].to(result_dtype)
                        if target_rows is not None:
                            target_position_mask[rows] = True
                        else:
                            processed_mask[rows] = True

            if target_rows is not None:
                fallback_positions = position_rows[valid_target_positions & (~target_position_mask)]
                fallback_rows = target_rows[fallback_positions]
            else:
                fallback_rows = (~processed_mask).nonzero(as_tuple=False).flatten()
            if fallback_rows.numel() > 0:
                fallback_logits = flat_logits[fallback_rows].to(dtype=result_dtype)
                fallback_labels = flat_labels[fallback_rows]
                fallback_label_logits = fallback_logits.gather(1, fallback_labels.unsqueeze(1)).squeeze(1)
                fallback_base_logprob = fallback_label_logits - torch.logsumexp(fallback_logits, dim=-1)
                result[fallback_rows] = fallback_base_logprob + log_one_minus_rho

            return result.reshape(labels.shape)

        fn.supports_target_rows = True
        return fn

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        calculate_varentropy: bool = False,
        calculate_shape_stats: bool = False,
        sampling_mask_logits_fn=None,
        entropy_stats_no_grad: bool = False,
        calculate_support_minp: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_varentropy is True:
                    varentropys: (bs, response_len)
                if calculate_shape_stats is True and not use_remove_padding:
                    h1 / h2: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            varentropy = None
            h1 = None
            h2 = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                topk_indices_rmpad = None
                keep_flags_rmpad = None
                sampled_logits_rmpad = None
                max_logits_rmpad = None
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                    if calculate_varentropy:
                        # Fused LM head does not expose Var(log pi); keep tensor for downstream shape only
                        varentropy_rmpad = torch.zeros_like(entropy_rmpad)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    if calculate_support_minp:
                        sampled_logits_rmpad = logits_rmpad.gather(
                            -1, input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                        max_logits_rmpad = logits_rmpad.max(dim=-1).values

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    _fused_entropy = False
                    if sampling_mask_logits_fn is not None:
                        target_rows = None
                        if getattr(sampling_mask_logits_fn, "supports_target_rows", False):
                            target_rows = self._compute_response_target_rows(
                                indices=indices,
                                batch_size=batch_size,
                                seqlen=seqlen,
                                response_length=response_length,
                            )
                        if hasattr(sampling_mask_logits_fn, '_compute_entropy'):
                            sampling_mask_logits_fn._compute_entropy = calculate_entropy
                            sampling_mask_logits_fn._compute_varentropy = calculate_varentropy
                            sampling_mask_logits_fn._shape_topk = self.config.get("shape_topk", 20)
                        if target_rows is not None:
                            log_probs = sampling_mask_logits_fn(
                                logits_rmpad,
                                input_ids_rmpad_rolled,
                                target_rows=target_rows,
                            )
                        else:
                            log_probs = sampling_mask_logits_fn(logits_rmpad, input_ids_rmpad_rolled)
                        if hasattr(sampling_mask_logits_fn, 'last_topk_indices'):
                            topk_indices_rmpad = sampling_mask_logits_fn.last_topk_indices
                        if hasattr(sampling_mask_logits_fn, 'last_keep_flags'):
                            keep_flags_rmpad = sampling_mask_logits_fn.last_keep_flags
                        if calculate_entropy and hasattr(sampling_mask_logits_fn, 'last_entropy'):
                            _fused_entropy = True
                            entropy_rmpad = sampling_mask_logits_fn.last_entropy
                            if calculate_varentropy and hasattr(sampling_mask_logits_fn, 'last_varentropy'):
                                varentropy_rmpad = sampling_mask_logits_fn.last_varentropy
                        if self.use_ulysses_sp and topk_indices_rmpad is not None:
                            from verl.utils.ulysses import get_ulysses_sequence_parallel_group
                            sp_group = get_ulysses_sequence_parallel_group()
                            if sp_group is not None:
                                local_k = torch.tensor([topk_indices_rmpad.shape[-1]],
                                                       device=topk_indices_rmpad.device)
                                torch.distributed.all_reduce(local_k, op=torch.distributed.ReduceOp.MAX,
                                                             group=sp_group)
                                global_max_k = int(local_k.item())
                                cur_k = topk_indices_rmpad.shape[-1]
                                if cur_k < global_max_k:
                                    topk_indices_rmpad = torch.nn.functional.pad(
                                        topk_indices_rmpad, (0, global_max_k - cur_k), value=0)
                                    if keep_flags_rmpad is not None:
                                        keep_flags_rmpad = torch.nn.functional.pad(
                                            keep_flags_rmpad, (0, global_max_k - cur_k), value=False)
                    else:
                        inplace_backward = True
                        if calculate_entropy:
                            inplace_backward = False
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy (skip if already fused into min-p chunked pass)
                    # Same code pattern as non-rmpad branch (verl_F.entropy_from_logits + log_softmax V)
                    if calculate_entropy and not _fused_entropy:
                        stats_logits = logits_rmpad.detach() if entropy_stats_no_grad else logits_rmpad
                        if calculate_varentropy:
                            entropy_rmpad, varentropy_rmpad = verl_F.entropy_and_varentropy_from_logits_logprob_domain(
                                stats_logits,
                                chunk_rows=self.config.get("entropy_varentropy_chunk_rows", 256),
                            )
                        else:
                            entropy_rmpad = self.compute_entropy_from_logits(stats_logits)

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                    # Free full-vocab logits early if entropy was fused and sum_pi_squared not needed
                    if _fused_entropy and not calculate_sum_pi_squared:
                        del output, logits_rmpad

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_varentropy:
                        varentropy_rmpad = gather_outputs_and_unpad(
                            varentropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if topk_indices_rmpad is not None:
                        topk_indices_rmpad = gather_outputs_and_unpad(
                            topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if keep_flags_rmpad is not None:
                        keep_flags_rmpad = gather_outputs_and_unpad(
                            keep_flags_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if sampled_logits_rmpad is not None:
                        sampled_logits_rmpad = gather_outputs_and_unpad(
                            sampled_logits_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if max_logits_rmpad is not None:
                        max_logits_rmpad = gather_outputs_and_unpad(
                            max_logits_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if calculate_varentropy:
                        varentropy_rmpad = varentropy_rmpad[:0]
                    if topk_indices_rmpad is not None:
                        topk_indices_rmpad = topk_indices_rmpad[:0]
                    if keep_flags_rmpad is not None:
                        keep_flags_rmpad = keep_flags_rmpad[:0]
                    if sampled_logits_rmpad is not None:
                        sampled_logits_rmpad = sampled_logits_rmpad[:0]
                    if max_logits_rmpad is not None:
                        max_logits_rmpad = max_logits_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_varentropy:
                    full_varentropy = pad_input(
                        hidden_states=varentropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size, seqlen=seqlen,
                    )

                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if topk_indices_rmpad is not None:
                    full_topk_indices = pad_input(
                        hidden_states=topk_indices_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if keep_flags_rmpad is not None:
                    full_keep_flags = pad_input(
                        hidden_states=keep_flags_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if sampled_logits_rmpad is not None:
                    full_sampled_logits = pad_input(
                        hidden_states=sampled_logits_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if max_logits_rmpad is not None:
                    full_max_logits = pad_input(
                        hidden_states=max_logits_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_varentropy:
                    varentropy = full_varentropy.squeeze(-1)[:, -response_length - 1 : -1]
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if topk_indices_rmpad is not None:
                    topk_indices_rmpad = full_topk_indices[:, -response_length - 1 : -1, :]
                if keep_flags_rmpad is not None:
                    keep_flags_rmpad = full_keep_flags[:, -response_length - 1 : -1, :]
                if sampled_logits_rmpad is not None:
                    sampled_logits_rmpad = full_sampled_logits.squeeze(-1)[:, -response_length - 1 : -1]
                if max_logits_rmpad is not None:
                    max_logits_rmpad = full_max_logits.squeeze(-1)[:, -response_length - 1 : -1]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                sampled_logits = None
                max_logits = None
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                    if calculate_varentropy:
                        varentropy = torch.zeros_like(entropy)

                else:
                    logits = output.logits
 
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if calculate_support_minp:
                        sampled_logits = logits.gather(-1, micro_batch["responses"].unsqueeze(-1)).squeeze(-1)
                        max_logits = logits.max(dim=-1).values
                    if sampling_mask_logits_fn is not None:
                        log_probs = sampling_mask_logits_fn(logits, micro_batch["responses"])
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        stats_logits = logits.detach() if entropy_stats_no_grad else logits
                        if calculate_varentropy:
                            entropy, varentropy = verl_F.entropy_and_varentropy_from_logits_logprob_domain(
                                stats_logits,
                                chunk_rows=self.config.get("entropy_varentropy_chunk_rows", 256),
                            )
                        else:
                            entropy = self.compute_entropy_from_logits(stats_logits)
                        if calculate_shape_stats:
                            # Shape credit only needs top-k Renyi summaries, not full varentropy.
                            _shape_K = self.config.get("shape_topk", 20)
                            topk_logits_ve, _ = torch.topk(logits, k=_shape_K, dim=-1)
                            topk_probs_ve = torch.softmax(topk_logits_ve, dim=-1)
                            topk_logp_ve = torch.log_softmax(topk_logits_ve, dim=-1)
                            topk_ent_ve = -(topk_probs_ve * topk_logp_ve).sum(dim=-1)
                            h1 = topk_ent_ve
                            h2 = -torch.log((topk_probs_ve ** 2).sum(dim=-1).clamp(min=1e-12))
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_varentropy:
                outputs["varentropys"] = varentropy
            if calculate_shape_stats and not self.use_remove_padding:
                outputs["h1"] = h1
                outputs["h2"] = h2
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if sampling_mask_logits_fn is not None and hasattr(sampling_mask_logits_fn, 'last_topk_indices'):
                if self.use_remove_padding:
                    if topk_indices_rmpad is not None:
                        outputs["topk_indices"] = topk_indices_rmpad
                else:
                    outputs["topk_indices"] = sampling_mask_logits_fn.last_topk_indices
            if sampling_mask_logits_fn is not None and hasattr(sampling_mask_logits_fn, 'last_keep_flags'):
                if self.use_remove_padding:
                    if keep_flags_rmpad is not None:
                        outputs["keep_flags"] = keep_flags_rmpad
                else:
                    outputs["keep_flags"] = sampling_mask_logits_fn.last_keep_flags
            if calculate_support_minp:
                if self.use_remove_padding:
                    if sampled_logits_rmpad is not None:
                        outputs["sampled_logits"] = sampled_logits_rmpad
                    if max_logits_rmpad is not None:
                        outputs["max_logits"] = max_logits_rmpad
                else:
                    outputs["sampled_logits"] = sampled_logits
                    outputs["max_logits"] = max_logits
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy: bool = False,
        calculate_varentropy: bool = False,
        calculate_shape_stats: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()
 
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
 
        # === Keep Sampling Mask (top-k or min-p) ===
        keep_sampling_top_k = self.config.get("keep_sampling_top_k", 0)
        keep_sampling_min_p = self.config.get("keep_sampling_min_p", 0.0)
        keep_sampling_feasible_set = self.config.get("keep_sampling_feasible_set", False)
        keep_sampling_feasible_set_t_in = self.config.get("keep_sampling_feasible_set_t_in", 1.0)
        keep_sampling_feasible_set_rho = self.config.get("keep_sampling_feasible_set_rho", 0.0)
        sampling_mask_logits_fn = None
        topk_indices_lst = []
        keep_flags_lst = []

        enabled_mask_modes = int(keep_sampling_top_k > 0) + int(keep_sampling_min_p > 0) + int(keep_sampling_feasible_set)
        if enabled_mask_modes > 1:
            raise ValueError(
                "keep_sampling_top_k, keep_sampling_min_p, and keep_sampling_feasible_set are mutually exclusive"
            )

        if (
            keep_sampling_feasible_set
            and "flat_topk_indices" in data.batch.keys()
            and "support_offsets" in data.batch.keys()
        ):
            _flat_topk_idx, _support_offsets = self._collapse_padded_flat_support(
                data.batch["flat_topk_indices"].to(torch.int64),
                data.batch["support_offsets"].to(torch.int64),
            )
            sampling_mask_logits_fn = self._make_rollout_feasible_set_mixture_logprob_fn(
                _flat_topk_idx,
                _support_offsets,
                t_in=keep_sampling_feasible_set_t_in,
                rho=keep_sampling_feasible_set_rho,
            )
        elif keep_sampling_feasible_set and "topk_indices" in data.batch.keys():
            _topk_idx = data.batch["topk_indices"].to(torch.int64)
            _keep_flags = data.batch.get("keep_flags", None)
            _support_counts = data.batch.get("support_counts", None)
            sampling_mask_logits_fn = self._make_rollout_sampling_mask_logits_fn(
                _topk_idx, _keep_flags, _support_counts
            )
        elif keep_sampling_top_k > 0:
            from verl.utils.torch_functional import build_topk_mask, logprobs_from_logits_masked

            def make_masked_logprob_fn_topk():
                def fn(logits, labels):
                    mask = build_topk_mask(logits, keep_sampling_top_k, response_ids=labels)
                    _, tidx = torch.topk(logits, k=keep_sampling_top_k, dim=-1)
                    fn.last_topk_indices = tidx
                    return logprobs_from_logits_masked(logits, labels, mask)
                return fn

            sampling_mask_logits_fn = make_masked_logprob_fn_topk()

        elif keep_sampling_min_p > 0:
            from verl.utils.torch_functional import fused_minp_logprob_chunked

            def make_masked_logprob_fn_minp():
                def fn(logits, labels):
                    result = fused_minp_logprob_chunked(
                        logits,
                        labels,
                        keep_sampling_min_p,
                        compute_entropy=fn._compute_entropy,
                        compute_varentropy=fn._compute_varentropy,
                        shape_topk=fn._shape_topk,
                    )
                    lp, tidx, flags = result[:3]
                    fn.last_topk_indices = tidx
                    fn.last_keep_flags = flags
                    if fn._compute_entropy and len(result) > 3:
                        fn.last_entropy = result[3]
                        if fn._compute_varentropy and len(result) > 4:
                            fn.last_varentropy = result[4]
                    return lp
                fn._compute_entropy = False
                fn._compute_varentropy = False
                fn._shape_topk = 20
                return fn

            sampling_mask_logits_fn = make_masked_logprob_fn_minp()
        # === end Keep Sampling Mask ===
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if keep_sampling_feasible_set and "flat_topk_indices" in data.batch.keys():
            select_keys.extend(["flat_topk_indices", "support_offsets"])
        elif keep_sampling_feasible_set and "topk_indices" in data.batch.keys():
            select_keys.append("topk_indices")
            if "support_counts" in data.batch.keys():
                select_keys.append("support_counts")
            if "keep_flags" in data.batch.keys():
                select_keys.append("keep_flags")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        varentropy_lst = []
        h1_lst = []
        h2_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    calculate_varentropy=calculate_varentropy,
                    calculate_shape_stats=calculate_shape_stats,
                    sampling_mask_logits_fn=sampling_mask_logits_fn,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_varentropy:
                varentropy_lst.append(outputs["varentropys"])
            if "h1" in outputs:
                h1_lst.append(outputs["h1"])
                h2_lst.append(outputs["h2"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])
            if "topk_indices" in outputs:
                topk_indices_lst.append(outputs["topk_indices"])
            if "keep_flags" in outputs:
                keep_flags_lst.append(outputs["keep_flags"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_varentropy:
            varentropys = torch.concat(varentropy_lst, dim=0)
        h1s = torch.concat(h1_lst, dim=0) if h1_lst else None
        h2s = torch.concat(h2_lst, dim=0) if h2_lst else None
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_varentropy:
                varentropys = restore_dynamic_batch(varentropys, batch_idx_list)
            if h1s is not None:
                h1s = restore_dynamic_batch(h1s, batch_idx_list)
                h2s = restore_dynamic_batch(h2s, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_varentropy:
            outputs["varentropys"] = varentropys
        if h1s is not None:
            outputs["h1"] = h1s
            outputs["h2"] = h2s
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        if topk_indices_lst:
            topk_indices_all = torch.cat(topk_indices_lst, dim=0)
            if use_dynamic_bsz:
                topk_indices_all = restore_dynamic_batch(topk_indices_all, batch_idx_list)
            outputs["topk_indices"] = topk_indices_all
            if keep_flags_lst:
                keep_flags_all = torch.cat(keep_flags_lst, dim=0)
                if use_dynamic_bsz:
                    keep_flags_all = restore_dynamic_batch(keep_flags_all, batch_idx_list)
                outputs["keep_flags"] = keep_flags_all
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if "flat_topk_indices" in data.batch.keys():
            select_keys.extend(["flat_topk_indices", "support_offsets"])
        elif "topk_indices" in data.batch.keys():
            select_keys.append("topk_indices")
        if "support_counts" in data.batch.keys():
            select_keys.append("support_counts")
        if "keep_flags" in data.batch.keys():
            select_keys.append("keep_flags")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "ref_log_prob" in data.batch.keys() and "ref_log_prob" not in select_keys:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    adv_reweight_enable = self.config.get("adv_reweight_enable", False)
                    beta_varentropy = self.config.get("beta_varentropy", 0.0)
                    loss_agg_mode = self.config.loss_agg_mode
                    stapo_enable = self.config.get("stapo_enable", False)
                    tampo_mask_enable = self.config.get("tampo_mask_enable", False)
                    use_shape_credit = self.config.get("use_shape_credit", False)
                    support_set_min_p = self.config.get("support_set_min_p", None)
                    taco_enable = self.config.get("taco_enable", False)
                    hard_mask_modes = int(stapo_enable) + int(tampo_mask_enable) + int(support_set_min_p is not None)
                    if hard_mask_modes > 1:
                        raise ValueError(
                            "stapo_enable, tampo_mask_enable, and support_set_min_p are mutually exclusive"
                        )
                    calculate_support_minp = (support_set_min_p is not None) and (not stapo_enable) and (not tampo_mask_enable)
                    if adv_reweight_enable:
                        _adv_alpha = float(self.config.get("adv_reweight_alpha", 0.1))
                        if not (0.0 <= _adv_alpha <= 1.0):
                            raise ValueError(
                                "adv_reweight_alpha must be in [0, 1], "
                                f"got {_adv_alpha}"
                            )
                    if stapo_enable:
                        _stapo_prob_threshold = float(self.config.get("stapo_prob_threshold", 0.002))
                        _stapo_entropy_quantile = float(self.config.get("stapo_entropy_quantile", 0.1))
                        if not (0.0 < _stapo_prob_threshold < 1.0):
                            raise ValueError(
                                f"stapo_prob_threshold must be in (0, 1), got {_stapo_prob_threshold}"
                            )
                        if not (0.0 < _stapo_entropy_quantile < 1.0):
                            raise ValueError(
                                f"stapo_entropy_quantile must be in (0, 1), got {_stapo_entropy_quantile}"
                            )
                    if calculate_support_minp:
                        if not (0.0 < float(support_set_min_p) < 1.0):
                            raise ValueError(f"support_set_min_p must be in (0, 1), got {support_set_min_p}")
                        if self.use_fused_kernels:
                            raise NotImplementedError("support_set_min_p is not supported with use_fused_kernels=True")
                    if taco_enable:
                        _alpha = float(self.config.get("taco_alpha", 0.01))
                        _lambda = float(self.config.get("taco_lambda", 0.9))
                        if _alpha <= 0.0:
                            raise ValueError(
                                "taco_alpha must be > 0, "
                                f"got {_alpha}"
                            )
                        if not (0.0 <= _lambda <= 1.0):
                            raise ValueError(
                                "taco_lambda must be in [0, 1], "
                                f"got {_lambda}"
                            )

                    calculate_entropy = (
                        self.config.calculate_entropy
                        or (entropy_coeff != 0)
                        or (beta_varentropy > 0)
                        or stapo_enable
                        or tampo_mask_enable
                        or taco_enable
                    )
                    calculate_varentropy = beta_varentropy > 0
                    entropy_stats_no_grad = calculate_entropy and entropy_coeff == 0 and beta_varentropy == 0

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    keep_sampling_top_k = self.config.get("keep_sampling_top_k", 0)
                    keep_sampling_min_p = self.config.get("keep_sampling_min_p", 0.0)
                    keep_sampling_feasible_set = self.config.get("keep_sampling_feasible_set", False)
                    keep_sampling_feasible_set_t_in = self.config.get("keep_sampling_feasible_set_t_in", 1.0)
                    keep_sampling_feasible_set_rho = self.config.get("keep_sampling_feasible_set_rho", 0.0)
                    sampling_mask_logits_fn = None
                    enabled_mask_modes = (
                        int(keep_sampling_top_k > 0) + int(keep_sampling_min_p > 0) + int(keep_sampling_feasible_set)
                    )
                    if enabled_mask_modes > 1:
                        raise ValueError(
                            "keep_sampling_top_k, keep_sampling_min_p, and keep_sampling_feasible_set "
                            "are mutually exclusive"
                        )

                    if (
                        keep_sampling_feasible_set
                        and "flat_topk_indices" in model_inputs
                        and "support_offsets" in model_inputs
                    ):
                        _flat_topk_idx, _support_offsets = self._collapse_padded_flat_support(
                            model_inputs["flat_topk_indices"].to(torch.int64),
                            model_inputs["support_offsets"].to(torch.int64),
                        )
                        sampling_mask_logits_fn = self._make_rollout_feasible_set_mixture_logprob_fn(
                            _flat_topk_idx,
                            _support_offsets,
                            t_in=keep_sampling_feasible_set_t_in,
                            rho=keep_sampling_feasible_set_rho,
                        )
                    elif keep_sampling_feasible_set and "topk_indices" in model_inputs:
                        _topk_idx = model_inputs["topk_indices"].to(torch.int64)  # (micro_bsz, response_len, max_k)
                        _keep_flags = model_inputs.get("keep_flags", None)  # (micro_bsz, response_len, max_k) bool or None
                        _support_counts = model_inputs.get("support_counts", None)  # (micro_bsz, response_len) or None
                        sampling_mask_logits_fn = self._make_rollout_sampling_mask_logits_fn(
                            _topk_idx, _keep_flags, _support_counts
                        )
                    elif (keep_sampling_top_k > 0 or keep_sampling_min_p > 0) and "topk_indices" in model_inputs:
                        _topk_idx = model_inputs["topk_indices"].to(torch.int64)  # (micro_bsz, response_len, max_k)
                        _keep_flags = model_inputs.get("keep_flags", None)  # (micro_bsz, response_len, max_k) bool or None
                        sampling_mask_logits_fn = self._make_rollout_sampling_mask_logits_fn(_topk_idx, _keep_flags)
                    # === end Keep Sampling Mask ===
 
                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        calculate_varentropy=calculate_varentropy,
                        calculate_shape_stats=False,
                        sampling_mask_logits_fn=sampling_mask_logits_fn,
                        entropy_stats_no_grad=entropy_stats_no_grad,
                        calculate_support_minp=calculate_support_minp,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    # === Shape-aware credit assignment ===
                    if use_shape_credit and calculate_entropy:
                        import math as _math
                        _shape_K = self.config.get("shape_topk", 20)
                        _clip_eps = self.config.clip_ratio_low
                        _log_K = _math.log(_shape_K)

                        _h1 = model_inputs.get("h1", None)
                        _h2 = model_inputs.get("h2", None)
                        _ve = model_inputs.get("varentropys", None)

                        if _h1 is not None and _h2 is not None:
                            _h1 = _h1.detach()
                            _h2 = _h2.detach()

                            # Rényi gap
                            _rg = (_h1 - _h2).clamp(min=0.0)

                            # Shape gate: ρ = (rg/h1) · 4·h_norm·(1-h_norm)
                            _r = _rg / (_h1 + 1e-6)
                            _rho = _r.pow(0.3) 

                            # Exact varentropy direction (analytic):
                            # sgn(∂V/∂ℓ_y) = sgn((a_y - (H+1))² - (V+1))
                            _a_y = -old_log_prob.detach()  # full-vocab -log p(y)
                            if _ve is not None:
                                _ve_d = _ve.detach()
                            else:
                                _ve_d = torch.zeros_like(_h1)
                            _d = torch.sign((_a_y - (_h1 + 1.0)).pow(2) - (_ve_d + 1.0))

                            # Direction consistency: s = ρ · (-sgn(A) · d)
                            _adv_sgn = torch.sign(advantages.detach())
                            _s = _rho * (-_adv_sgn * _d)

                            # Bounded base weight
                            _w_base = 1.0 + _clip_eps * _s

                            # Asymmetric p(1-p) brake on amplification only
                            _p_y = old_log_prob.detach().exp().clamp(1e-8, 1.0 - 1e-8)
                            _alpha = 0.5 * (1.0 - _adv_sgn * (2.0 * _p_y - 1.0))
                            _w = 1.0 + (_w_base - 1.0) * _alpha         # 改动 3
                            _w = _w.clamp(1.0 - _clip_eps, 1.0 + _clip_eps)

                            # Modulate advantages
                            advantages = advantages * _w

                            # Metrics
                            _rm = response_mask
                            _rm_sum = _rm.sum().clamp(min=1)
                            micro_batch_metrics["shape/gate_rho"] = (_rho * _rm).sum().item() / _rm_sum.item()
                            micro_batch_metrics["shape/renyi_gap"] = (_rg * _rm).sum().item() / _rm_sum.item()
                            micro_batch_metrics["shape/direction_s"] = (_s * _rm).sum().item() / _rm_sum.item()
                            micro_batch_metrics["shape/weight_mean"] = (_w * _rm).sum().item() / _rm_sum.item()
                    # === end shape-aware credit ===
                    # === AR-Lopti style advantage reweighting ===
                    if adv_reweight_enable:
                        _old_prob = old_log_prob.detach().float().exp()
                        _adv_weight = (1.0 - _adv_alpha) + _adv_alpha * _old_prob
                        advantages = advantages * _adv_weight.to(dtype=advantages.dtype)

                        _rm = response_mask
                        _rm_sum = _rm.sum().clamp(min=1)
                        micro_batch_metrics["adv_reweight/alpha"] = _adv_alpha
                        micro_batch_metrics["adv_reweight/weight_mean"] = (
                            (_adv_weight * _rm).sum().item() / _rm_sum.item()
                        )
                        micro_batch_metrics["adv_reweight/prob_mean"] = (
                            (_old_prob * _rm).sum().item() / _rm_sum.item()
                        )
                    else:
                        micro_batch_metrics["adv_reweight/alpha"] = 0.0
                        micro_batch_metrics["adv_reweight/weight_mean"] = 1.0
                        micro_batch_metrics["adv_reweight/prob_mean"] = 0.0
                    if taco_enable and entropy is not None:
                        advantages, taco_metrics = compute_taco_token_advantages(
                            log_prob=log_prob.detach().float(),
                            entropy=entropy.detach().float(),
                            advantages=advantages,
                            response_mask=response_mask,
                            alpha=float(self.config.get("taco_alpha", 0.01)),
                            lambda_=float(self.config.get("taco_lambda", 0.9)),
                        )
                        for key, value in taco_metrics.items():
                            micro_batch_metrics[f"taco/{key}"] = value.item()
                    # === STAPO-style hard masking on positive-advantage tokens ===
                    if stapo_enable and entropy is not None:
                        _cur_lp = log_prob.detach().float()
                        _cur_prob = _cur_lp.exp()
                        _cur_entropy = entropy.detach().float()
                        _adv = advantages.detach()
                        _rm = response_mask
                        _rm_sum = _rm.sum().clamp(min=1)
                        _valid_gate = _rm > 0
                        _only_pos = self.config.get("stapo_only_positive_advantage", True)
                        if _only_pos:
                            _candidate_gate = _valid_gate & (_adv > 0)
                        else:
                            _candidate_gate = _valid_gate

                        _candidate_sum = _candidate_gate.sum()
                        _prob_threshold = float(self.config.get("stapo_prob_threshold", 0.002))
                        _entropy_quantile = float(self.config.get("stapo_entropy_quantile", 0.1))
                        _entropy_source = _cur_entropy[_candidate_gate]
                        if _entropy_source.numel() == 0:
                            _entropy_source = _cur_entropy[_valid_gate]

                        if _entropy_source.numel() > 0:
                            _entropy_threshold = torch.quantile(_entropy_source, _entropy_quantile).item()
                        else:
                            _entropy_threshold = 0.0

                        _spurious_gate = (
                            _candidate_gate
                            & (_cur_prob < _prob_threshold)
                            & (_cur_entropy < _entropy_threshold)
                        )
                        _stapo_weight = torch.where(
                            _spurious_gate,
                            torch.zeros_like(_cur_prob),
                            torch.ones_like(_cur_prob),
                        )
                        _masked_sum = _spurious_gate.sum()

                        micro_batch_metrics["stapo/prob_threshold"] = _prob_threshold
                        micro_batch_metrics["stapo/entropy_threshold"] = _entropy_threshold
                        micro_batch_metrics["stapo/weight_mean"] = (
                            (_stapo_weight * _rm).sum().item() / _rm_sum.item()
                        )
                        micro_batch_metrics["stapo/active_ratio"] = _candidate_sum.item() / _rm_sum.item()
                        if _candidate_sum.item() > 0:
                            _candidate_float = _candidate_gate.float()
                            micro_batch_metrics["stapo/in_ratio"] = (
                                ((_stapo_weight > 0.5) & _candidate_gate).float().sum().item() / _candidate_sum.item()
                            )
                            micro_batch_metrics["stapo/masked_ratio"] = _masked_sum.item() / _candidate_sum.item()
                            micro_batch_metrics["stapo/prob_mean"] = (
                                (_cur_prob * _candidate_float).sum().item() / _candidate_sum.item()
                            )
                            micro_batch_metrics["stapo/entropy_mean"] = (
                                (_cur_entropy * _candidate_float).sum().item() / _candidate_sum.item()
                            )
                        else:
                            micro_batch_metrics["stapo/in_ratio"] = 0.0
                            micro_batch_metrics["stapo/masked_ratio"] = 0.0
                            micro_batch_metrics["stapo/prob_mean"] = 0.0
                            micro_batch_metrics["stapo/entropy_mean"] = 0.0
                        if _masked_sum.item() > 0:
                            _filtered_probs = _cur_prob[_spurious_gate]
                            micro_batch_metrics["stapo/filtered_max_prob"] = _filtered_probs.max().item()
                            micro_batch_metrics["stapo/filtered_mean_prob"] = _filtered_probs.mean().item()
                        else:
                            micro_batch_metrics["stapo/filtered_max_prob"] = 0.0
                            micro_batch_metrics["stapo/filtered_mean_prob"] = 0.0
                        response_mask = response_mask * _stapo_weight
                    # === TAMPO-style hard truncation on positive-advantage tokens ===
                    elif tampo_mask_enable:
                        _cur_lp = log_prob.detach().float()
                        _cur_prob = _cur_lp.exp()
                        _cur_entropy = outputs["entropys"].detach().float()
                        _adv = advantages.detach()
                        _rm = response_mask
                        _rm_sum = _rm.sum().clamp(min=1)
                        _pos_gate = (_rm > 0) & (_adv > 0)
                        _pos_sum = _pos_gate.sum()
                        _prob_threshold = float(self.config.get("tampo_prob_threshold", 0.002))
                        _entropy_threshold = float(self.config.get("tampo_entropy_threshold", 0.5))
                        _truncate_gate = _pos_gate & (_cur_prob <= _prob_threshold) & (_cur_entropy <= _entropy_threshold)
                        _support_weight = torch.where(_truncate_gate, torch.zeros_like(_cur_prob), torch.ones_like(_cur_prob))
                        if _pos_sum.item() > 0:
                            _pos_float = _pos_gate.float()
                            micro_batch_metrics["support_set/in_ratio"] = (
                                ((_support_weight > 0.5) & _pos_gate).float().sum().item() / _pos_sum.item()
                            )
                            micro_batch_metrics["support_set/prob_mean"] = (
                                (_cur_prob * _pos_float).sum().item() / _pos_sum.item()
                            )
                            micro_batch_metrics["support_set/entropy_mean"] = (
                                (_cur_entropy * _pos_float).sum().item() / _pos_sum.item()
                            )
                        else:
                            micro_batch_metrics["support_set/in_ratio"] = 0.0
                            micro_batch_metrics["support_set/prob_mean"] = 0.0
                            micro_batch_metrics["support_set/entropy_mean"] = 0.0
                        micro_batch_metrics["support_set/weight_mean"] = (
                            (_support_weight * _rm).sum().item() / _rm_sum.item()
                        )
                        if _truncate_gate.any().item():
                            _filtered_probs = _cur_prob[_truncate_gate]
                            micro_batch_metrics["support_set/filtered_max_prob"] = _filtered_probs.max().item()
                            micro_batch_metrics["support_set/filtered_mean_prob"] = _filtered_probs.mean().item()
                        else:
                            micro_batch_metrics["support_set/filtered_max_prob"] = 0.0
                            micro_batch_metrics["support_set/filtered_mean_prob"] = 0.0
                        response_mask = response_mask * _support_weight
                    # === Min-p inspired support weighting on current policy logits ===
                    # Tokens inside the min-p set keep full weight 1.0.
                    # Tokens outside decay smoothly toward support_set_min_weight.
                    elif calculate_support_minp:
                        _cur_lp = log_prob.detach().float()
                        _sampled_logits = outputs["sampled_logits"].detach().float()
                        _max_logits = outputs["max_logits"].detach().float()
                        _gap = _sampled_logits - _max_logits
                        _min_p_value = float(support_set_min_p)
                        _log_min_p = torch.log(
                            torch.tensor(_min_p_value, device=_gap.device, dtype=_gap.dtype)
                        )
                        _margin = _gap - _log_min_p
                        _in_support = _margin >= 0
                        _only_pos = self.config.get("support_set_only_positive_advantage", True)
                        _soft_tau = max(float(self.config.get("support_set_minp_tau", 0.2)), 1e-6)
                        _min_weight = float(self.config.get("support_set_min_weight", 0.2))
                        _min_weight = min(max(_min_weight, 0.0), 1.0)
                        _outside_margin = torch.clamp(_margin, max=0.0)
                        _support_weight = _min_weight + (1.0 - _min_weight) * torch.exp(_outside_margin / _soft_tau)
                        if _only_pos:
                            _adv = advantages.detach()
                            _support_weight = torch.where(
                                _adv > 0, _support_weight, torch.ones_like(_support_weight)
                            )
                        _rm = response_mask
                        _rm_sum = _rm.sum().clamp(min=1)
                        _pos_gate = (_rm > 0) & (advantages.detach() > 0)
                        _pos_sum = _pos_gate.sum()
                        _effective_hard_mask = _in_support.float()
                        if _only_pos:
                            _effective_hard_mask = torch.where(
                                advantages.detach() > 0, _effective_hard_mask, torch.ones_like(_effective_hard_mask)
                            )
                        _filtered_gate = (_rm > 0) & (_effective_hard_mask < 0.5)
                        _filtered_sum = _filtered_gate.sum()
                        if _pos_sum.item() > 0:
                            micro_batch_metrics["support_set/in_ratio"] = (
                                (_in_support & _pos_gate).float().sum().item() / _pos_sum.item()
                            )
                            micro_batch_metrics["support_set/gap_mean"] = (
                                (_gap * _pos_gate.float()).sum().item() / _pos_sum.item()
                            )
                            micro_batch_metrics["support_set/margin_mean"] = (
                                (_margin * _pos_gate.float()).sum().item() / _pos_sum.item()
                            )
                        else:
                            micro_batch_metrics["support_set/in_ratio"] = 0.0
                            micro_batch_metrics["support_set/gap_mean"] = 0.0
                            micro_batch_metrics["support_set/margin_mean"] = 0.0
                        micro_batch_metrics["support_set/weight_mean"] = (
                            (_support_weight * _rm).sum().item() / _rm_sum.item()
                        )
                        if _filtered_sum.item() > 0:
                            _filtered_probs = _cur_lp.exp()[_filtered_gate]
                            micro_batch_metrics["support_set/filtered_max_prob"] = _filtered_probs.max().item()
                            micro_batch_metrics["support_set/filtered_mean_prob"] = _filtered_probs.mean().item()
                        else:
                            micro_batch_metrics["support_set/filtered_max_prob"] = 0.0
                            micro_batch_metrics["support_set/filtered_mean_prob"] = 0.0
                        response_mask = response_mask * _support_weight
                    # === end support set mask ===

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    
                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff
                    # === varentropy loss ===
                    if beta_varentropy > 0 and calculate_entropy and "varentropys" in outputs:
                        varentropy_agg = agg_loss(
                            loss_mat=outputs["varentropys"],
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + beta_varentropy * varentropy_agg
                        micro_batch_metrics["actor/varentropy_loss"] = varentropy_agg.detach().item()
                    # === end varentropy loss ===
                    # === end varentropy loss ===
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
