################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import torch
import torch.distributed as dist
import dataclasses
import os
import time
from typing import Optional
import flash_comm._C.ep_intranode as _ep
import flash_comm._C.ep_internode as _ep_inter
import flash_comm._C.buffer as _buffer

from .ep_context import EPContext

# Single source of truth lives in C++ (flash_comm/ep/internode.h) and is
# exported through pybind; do not hardcode these values in Python.
_MAX_DISPATCH_PIPELINE_CHUNKS = int(_ep_inter.MAX_DISPATCH_PIPELINE_CHUNKS)
_MAX_COMBINE_PIPELINE_CHUNKS = int(_ep_inter.MAX_COMBINE_PIPELINE_CHUNKS)


def _optional_positive_int_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {parsed}")
    return parsed


def _resolve_pipeline_chunks(name: str, max_value: int) -> int:
    chunks = _optional_positive_int_env(name)
    if chunks is None:
        return 1
    if chunks > max_value:
        raise ValueError(f"{name} must be in [1, {max_value}], got {chunks}")
    return chunks


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclasses.dataclass
class EPCommLayoutDesc:
    # only dependent on local topk_indices, can be computed in advance
    token_within_expert_offset: Optional[torch.Tensor] = None  # [num_tokens, topk]
    expert_counts: Optional[torch.Tensor] = None  # [num_experts + 1]

    # dispatch layout
    recv_base_offset: Optional[torch.Tensor] = None  # [world_size, experts_per_rank, world_size]

    # token_dst_scatter_indices / recv_topk_scatter_indices both store "positions in the dispatch output buffer".
    #
    # - token_dst_scatter_indices: sender-token space mapping.
    #   Shape: [num_token, topk] (intranode).
    #   For each local input token t and its k-th expert choice:
    #     token_dst_scatter_indices[t, k] is the slot index inside the *target rank's* dispatch receive buffer
    #     where this (token, expert-choice) should be written during dispatch.
    #
    # - recv_topk_scatter_indices: receiver-token space mapping.
    #   Shape: [num_recv_token, topk] (intranode), where num_recv_token = recv_token_count[rank].
    #   For each token row r in *this rank's* dispatch receive buffer and its k-th lane:
    #     recv_topk_scatter_indices[r, k] stores the "scatter index" used by dispatch_postprocess/combine to
    #     relate received rows back to the original token ordering (and/or to compute dispatch_weights).
    #
    # In short:
    # - token_dst_scatter_indices is indexed by *sender-side* token rows (pre-dispatch).
    # - recv_topk_scatter_indices is indexed by *receiver-side* token rows (post-dispatch).
    # Both refer to offsets/slots in the dispatch receive buffers, but their index space (and thus shape) differs.
    token_dst_scatter_indices: Optional[
        torch.Tensor] = None  # intranode: [num_token, topk]; internode: [nnodes, max_tokens, topk]
    token_topk_send_mask: Optional[
        torch.Tensor] = None  # intranode: [num_token, topk]; internode: [nnodes, max_tokens, topk]
    topk_indices: Optional[torch.Tensor] = None  # intranode: [num_token, topk]; internode: [nnodes, max_tokens, topk]
    recv_token_count_cpu: Optional[torch.Tensor] = None  # [world_size] pinned CPU memory (unaligned)
    recv_token_count: Optional[torch.Tensor] = None  # [world_size] device memory (unaligned)
    # Internode-only: per-rank source token counts written by compute_dispatch_layout.
    # This is layout data for the current dispatch, not persistent communication state.
    num_tokens_per_rank: Optional[torch.Tensor] = None  # [world_size] device memory
    recv_aligned_token_count_cpu: Optional[torch.Tensor] = None  # [world_size] pinned CPU (aligned, for buffer alloc)
    recv_aligned_token_count: Optional[torch.Tensor] = None  # [world_size] device (aligned, for postprocess/combine)
    recv_expert_counts: Optional[torch.Tensor] = None  # [experts_per_rank] per-expert actual token counts
    expert_alignment: int = 1
    num_tokens: int = -1
    recv_topk_scatter_indices: Optional[
        torch.Tensor] = None  # intranode: [num_recv_token, topk], internode: [nnodes, max_tokens, topk]
    # Optional receiver-view metadata filled by compute_dispatch_layout for
    # CuTeDSL pull dispatch / push combine overlap paths.
    token_src_rank_topk_and_indices: Optional[torch.Tensor] = None  # [num_recv_token] int64

    def check_combine_required_inputs(self):
        if self.token_topk_send_mask is None or self.token_dst_scatter_indices is None:
            raise ValueError("token_topk_send_mask and token_dst_scatter_indices must be provided")

    def need_recompute_token_within_expert_offset_and_expert_counts(self, topk_indices: Optional[torch.Tensor] = None):
        if self.token_within_expert_offset is None or self.expert_counts is None:
            return True
        if topk_indices is not None and tuple(self.token_within_expert_offset.shape) != tuple(topk_indices.shape):
            return True
        return False

    def need_recompute_dispatch_layout(self, expert_alignment: int = 1, num_tokens: int = -1):
        if (self.recv_base_offset is None or self.token_dst_scatter_indices is None or self.token_topk_send_mask is None
                or self.recv_token_count_cpu is None or self.recv_token_count is None
                or self.recv_expert_counts is None):
            return True
        if self.expert_alignment != expert_alignment:
            return True
        if num_tokens >= 0 and self.num_tokens != num_tokens:
            return True
        if expert_alignment > 1:
            if self.recv_aligned_token_count_cpu is None or self.recv_aligned_token_count is None:
                return True
        return False

    def check_layout_desc(
        self,
        num_tokens: int,
        topk: int,
        num_experts: int,
        world_size: int = 1,
        local_world_size: Optional[int] = None,
        max_slot_num_token: int = -1,
    ):
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if num_experts <= 0 or num_experts % world_size != 0:
            raise ValueError(f"num_experts must be positive and divisible by world_size, got "
                             f"num_experts={num_experts}, world_size={world_size}")
        if local_world_size is None:
            local_world_size = world_size
        if local_world_size <= 0 or world_size % local_world_size != 0:
            raise ValueError(f"local_world_size must divide world_size, got world_size={world_size}, "
                             f"local_world_size={local_world_size}")
        nnodes = world_size // local_world_size
        layout_num_tokens = self.num_tokens if self.num_tokens >= 0 else num_tokens

        def internode_metadata_shape(name: str):
            if max_slot_num_token <= 0:
                raise ValueError(f"max_slot_num_token must be positive when validating rank-3 {name}, "
                                 f"got {max_slot_num_token}")
            return (nnodes, max_slot_num_token, topk)

        def check_shape(tensor: Optional[torch.Tensor], name: str, expected_shape):
            if tensor is not None and tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(f"{name} must have shape {list(expected_shape)}, got shape {tuple(tensor.shape)}")

        def check_token_topk_shape(tensor: Optional[torch.Tensor], name: str):
            if tensor is None:
                return
            if tensor.dim() == 2:
                if nnodes > 1:
                    raise ValueError(f"{name} must be rank-3 for internode layout, got shape {tuple(tensor.shape)}")
                expected = (layout_num_tokens, topk)
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"{name} must have shape {list(expected)}, got shape {tuple(tensor.shape)}")
            elif tensor.dim() == 3:
                if nnodes <= 1:
                    raise ValueError(f"{name} must be rank-2 for intranode layout, got shape {tuple(tensor.shape)}")
                expected = internode_metadata_shape(name)
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"{name} must have shape {list(expected)} for internode layout, "
                                     f"got shape {tuple(tensor.shape)}")
            else:
                raise ValueError(f"{name} must be rank-2 or rank-3, got shape {tuple(tensor.shape)}")

        def check_recv_topk_scatter_shape(tensor: Optional[torch.Tensor], name: str):
            if tensor is None:
                return
            if tensor.dim() == 2:
                if tensor.shape[1] != topk:
                    raise ValueError(f"{name} second dim must be topk={topk}, got shape {tuple(tensor.shape)}")
            elif tensor.dim() == 3:
                if nnodes <= 1:
                    raise ValueError(f"{name} must be rank-2 for intranode layout, got shape {tuple(tensor.shape)}")
                expected = internode_metadata_shape(name)
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"{name} must have shape {list(expected)} for internode layout, "
                                     f"got shape {tuple(tensor.shape)}")
            else:
                raise ValueError(f"{name} must be rank-2 or rank-3, got shape {tuple(tensor.shape)}")

        has_token_offset = self.token_within_expert_offset is not None
        has_expert_counts = self.expert_counts is not None
        if has_expert_counts != has_token_offset:
            raise ValueError("token_within_expert_offset and expert_counts must both be None or both be non-None")

        experts_per_rank = num_experts // world_size
        check_shape(self.token_within_expert_offset, "token_within_expert_offset", (layout_num_tokens, topk))
        check_shape(self.expert_counts, "expert_counts", (num_experts + 1, ))
        check_shape(self.recv_base_offset, "recv_base_offset", (world_size, experts_per_rank, world_size))
        check_token_topk_shape(self.token_dst_scatter_indices, "token_dst_scatter_indices")
        check_token_topk_shape(self.token_topk_send_mask, "token_topk_send_mask")
        check_token_topk_shape(self.topk_indices, "topk_indices")
        check_shape(self.recv_token_count_cpu, "recv_token_count_cpu", (world_size, ))
        check_shape(self.recv_token_count, "recv_token_count", (world_size, ))
        check_shape(self.recv_aligned_token_count_cpu, "recv_aligned_token_count_cpu", (world_size, ))
        check_shape(self.recv_aligned_token_count, "recv_aligned_token_count", (world_size, ))
        check_shape(self.recv_expert_counts, "recv_expert_counts", (experts_per_rank, ))
        check_shape(self.num_tokens_per_rank, "num_tokens_per_rank", (world_size, ))
        check_recv_topk_scatter_shape(self.recv_topk_scatter_indices, "recv_topk_scatter_indices")


class EPKernels:
    """
    FlashComm EP kernels.

    Internode EP owns one process-global NCCL GIN communicator and one RDMA rail
    buffer per EPKernels instance.  Dispatch/combine calls must be serialized;
    overlapping calls would race on those protocol slots.
    """

    def __init__(self, max_m: int, hidden: int, topk: int, num_experts: int, local_world_size: int,
                 ep_group: torch.distributed.ProcessGroup, num_sm: int = 16, capacity: float = 1.2,
                 num_worst_tokens: int = -1, expert_alignment: int = 1, check_num_worst_tokens: bool = False):
        self.ep_group = ep_group
        self.num_sm = num_sm
        self.alignment = 1024
        self.coeff = capacity  # for output buffer reallocation
        self.cpu_default_val = -1
        cpu_poll_sleep_us = int(os.environ.get("FLASH_COMM_EP_CPU_POLL_SLEEP_US", "50"))
        assert cpu_poll_sleep_us >= 0, f"cpu_poll_sleep_us must be >= 0, got {cpu_poll_sleep_us}"
        self.cpu_poll_sleep_us = cpu_poll_sleep_us
        self.cpu_poll_sleep_s = cpu_poll_sleep_us / 1_000_000
        self.rank = ep_group.rank()
        self.world_size = ep_group.size()
        self.local_world_size = local_world_size
        self.num_worst_tokens = num_worst_tokens
        assert expert_alignment >= 1, f"expert_alignment must be >= 1, got {expert_alignment}"
        self.expert_alignment = expert_alignment
        self.check_num_worst_tokens = check_num_worst_tokens

        self.is_internode = self.world_size > local_world_size
        if self.is_internode:
            self._init_internode_nccl_gin()

        self.ep_context = EPContext.create(
            max_m=max_m,
            hidden=hidden,
            topk=topk,
            num_experts=num_experts,
            group=ep_group,
            local_world_size=local_world_size,
            capacity_coeff=capacity,
            num_worst_tokens=num_worst_tokens,
        )
        torch.distributed.barrier(group=ep_group)

    def _init_internode_nccl_gin(self) -> None:
        ep_num_qps = _optional_positive_int_env("FLASH_COMM_EP_NUM_QPS") or 1
        gin_contexts = ep_num_qps

        root_global_rank = torch.distributed.get_global_rank(self.ep_group, 0)
        uid = [_buffer.nccl_gin_get_unique_id() if self.rank == 0 else None]
        dist.broadcast_object_list(uid, src=root_global_rank, group=self.ep_group)
        nnodes = self.world_size // self.local_world_size
        root_ep_num_qps = [ep_num_qps if self.rank == 0 else None]
        dist.broadcast_object_list(root_ep_num_qps, src=root_global_rank, group=self.ep_group)
        if ep_num_qps != root_ep_num_qps[0]:
            raise ValueError("FLASH_COMM_EP_NUM_QPS must be identical across the EP group: "
                             f"rank {self.rank} has {ep_num_qps}, root has {root_ep_num_qps[0]}")
        dispatch_chunks = _resolve_pipeline_chunks("FLASH_COMM_EP_DISPATCH_PIPELINE_CHUNKS",
                                                   _MAX_DISPATCH_PIPELINE_CHUNKS)
        combine_chunks = _resolve_pipeline_chunks("FLASH_COMM_EP_COMBINE_PIPELINE_CHUNKS", _MAX_COMBINE_PIPELINE_CHUNKS)
        root_chunks = [(dispatch_chunks, combine_chunks) if self.rank == 0 else None]
        dist.broadcast_object_list(root_chunks, src=root_global_rank, group=self.ep_group)
        if (dispatch_chunks, combine_chunks) != root_chunks[0]:
            raise ValueError("FLASH_COMM_EP_DISPATCH_PIPELINE_CHUNKS and "
                             "FLASH_COMM_EP_COMBINE_PIPELINE_CHUNKS must be identical "
                             "across the EP group: "
                             f"rank {self.rank} has {(dispatch_chunks, combine_chunks)}, "
                             f"root has {root_chunks[0]}")
        gin_signals = max(
            16,
            _round_up(int(_ep_inter.ep_required_gin_signal_count(nnodes)), 16),
        )
        gin_rail_barriers = max(16, self.num_sm)
        gin_queue_depth = 4096
        _buffer.nccl_gin_init(uid[0], self.rank, self.world_size, self.local_world_size,
                              gin_contexts, gin_signals, gin_rail_barriers, gin_queue_depth,
                              int(_buffer.NCCL_GIN_CONNECTION_FULL), ep_num_qps=ep_num_qps)
        # Context-local EP GIN signal id ranges (see flash_comm/ep/internode.h).
        # The fused reset+barrier after each dispatch/combine resets its
        # protocol range on every context.
        self._dispatch_signal_range = (0, int(_ep_inter.ep_dispatch_signal_count(nnodes)))
        self._combine_signal_range = (int(_ep_inter.ep_combine_signal_base(nnodes)),
                                      int(_ep_inter.ep_required_gin_signal_count(nnodes)))
        lsa_size = _buffer.nccl_gin_lsa_size()
        lsa_rank = _buffer.nccl_gin_lsa_rank()
        if lsa_size < self.local_world_size or (lsa_rank % self.local_world_size) != (self.rank %
                                                                                      self.local_world_size):
            raise ValueError("NCCL GIN LSA team must cover local_world_size/local_rank")

    def finalize(self) -> None:
        ctx = getattr(self, "ep_context", None)
        if self.is_internode:
            if ctx is not None:
                self.ep_context.release_internode_nccl_resources()
            torch.cuda.synchronize()
            torch.distributed.barrier(group=self.ep_group)
            if _buffer.nccl_gin_is_initialized():
                _buffer.nccl_gin_destroy()
            torch.cuda.synchronize()
            torch.distributed.barrier(group=self.ep_group)
        # Coordinated, leak-free release of the symmetric buffers. Critical for the
        # torch_ipc backend so producer storage is not left in torch's CUDA IPC
        # limbo across EPKernels re-creation. Must be called collectively by every
        # rank -- this is why teardown is an explicit finalize() and not __del__.
        if ctx is not None:
            ctx.free_buffers()

    def _realloc_dispatch_output_buf(self, recv_token_count_cpu: torch.Tensor, recv_token_count: torch.Tensor):
        assert recv_token_count_cpu.is_cpu
        assert recv_token_count_cpu.dtype == torch.int32
        max_output_token_num = 0

        if self.num_worst_tokens > 0 and self.num_worst_tokens <= self.ep_context.dispatch_output_buf.shape[0]:
            if self.check_num_worst_tokens:
                torch._assert_async(recv_token_count[self.rank] <= self.num_worst_tokens,
                                    f"num_worst_tokens = {self.num_worst_tokens} is not valid")
            return self.num_worst_tokens, self.num_worst_tokens

        # for target_rank in range(self.ep_context.config.world_size):
        #     # slice and item operations of the tensor are too time-consuming (10us level), so here we read directly from ptr
        #     while ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value == self.cpu_default_val:
        #         pass
        #     cur_output_token_num = ctypes.c_int32.from_address(base_ptr + target_rank * elem_size).value
        #     max_output_token_num = max(max_output_token_num, cur_output_token_num)
        arr = recv_token_count_cpu.numpy()
        while int(arr.min()) == self.cpu_default_val:
            if self.cpu_poll_sleep_us > 0:
                time.sleep(self.cpu_poll_sleep_s)
        max_output_token_num = int(arr.max())
        cur_output_token_num = int(arr[self.rank])
        if max_output_token_num > self.ep_context.dispatch_output_buf.shape[0]:
            self.ep_group_barrier()
            torch.cuda.synchronize()

            alloc_token = int(
                (max_output_token_num + self.alignment - 1) // self.alignment * self.alignment * self.coeff)
            print(
                f"reallocate dispatch output buf from {self.ep_context.dispatch_output_buf.shape[0]} to {alloc_token}")
            self.ep_context.reallocate_buffers(alloc_token)
            self.ep_group_barrier()
            torch.cuda.synchronize()

        return cur_output_token_num, max_output_token_num

    def dispatch_intranode(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: Optional[torch.Tensor],
                           layout_desc: EPCommLayoutDesc):
        self.ep_group_barrier()
        # recompute if not provided
        if layout_desc.need_recompute_token_within_expert_offset_and_expert_counts(topk_indices):
            layout_desc.token_within_expert_offset, layout_desc.expert_counts = \
                self.compute_stable_local_token_within_expert_offset_and_expert_counts(topk_indices, self.num_sm)

        num_token = input.shape[0]
        if layout_desc.need_recompute_dispatch_layout(self.expert_alignment, num_token):
            (
                layout_desc.recv_base_offset,
                layout_desc.token_dst_scatter_indices,
                layout_desc.token_topk_send_mask,
                layout_desc.recv_token_count_cpu,
                layout_desc.recv_token_count,
                layout_desc.recv_aligned_token_count_cpu,
                layout_desc.recv_aligned_token_count,
                layout_desc.recv_expert_counts,
            ) = _ep.compute_dispatch_layout(
                topk_indices,
                layout_desc.token_within_expert_offset,
                layout_desc.expert_counts,
                self.ep_context.full_splits_buf_ptrs,
                self.ep_context.nvl_barrier_buf_ptrs,
                self.ep_context.config.num_experts,
                self.ep_context.config.rank,
                self.ep_context.config.world_size,
                self.num_sm,
                self.ep_context.recv_token_count_cpu,
                token_src_rank_topk_and_indices_ptrs=None,
                expert_alignment=self.expert_alignment,
            )
            layout_desc.expert_alignment = self.expert_alignment
            layout_desc.num_tokens = num_token

        self.ep_group_barrier()
        if layout_desc.expert_alignment > 1:
            assert layout_desc.recv_aligned_token_count_cpu is not None, \
                "recv_aligned_token_count_cpu must be set when expert_alignment > 1"
            buf_count_cpu = layout_desc.recv_aligned_token_count_cpu
            buf_count_gpu = layout_desc.recv_aligned_token_count
        else:
            buf_count_cpu = layout_desc.recv_token_count_cpu
            buf_count_gpu = layout_desc.recv_token_count
        dispatch_recv_token_count, _ = self._realloc_dispatch_output_buf(buf_count_cpu, buf_count_gpu)

        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        # dispatch
        # Note: if topk_weights is None, the kernel skips weight dispatch and the weight buffer contents are stale.
        _ep.dispatch_intranode(input, layout_desc.token_topk_send_mask, topk_weights, topk_indices,
                               layout_desc.token_dst_scatter_indices, self.ep_context.dispatch_output_buf_ptrs,
                               self.ep_context.dispatch_topk_weights_buf_ptrs,
                               self.ep_context.dispatch_topk_scatter_indices_buf_ptrs, self.rank, self.world_size,
                               num_experts_per_rank, self.num_sm)

        # push mode, need barrier on ep group, wait ep ranks to finish dispatch
        self.ep_group_barrier()
        layout_desc.topk_indices = topk_indices
        layout_desc.recv_topk_scatter_indices = self.ep_context.dispatch_topk_scatter_indices_buf[:
                                                                                                  dispatch_recv_token_count]
        # directly return the symm tensors to avoid extra copy
        dispatch_weights = None
        if topk_weights is not None:
            dispatch_weights = self.ep_context.dispatch_topk_weights_buf[:dispatch_recv_token_count]
        return (self.ep_context.dispatch_output_buf[:dispatch_recv_token_count], dispatch_weights, layout_desc)

    def dispatch_intranode_postprocess(self, dispatch_out: torch.Tensor, dispatch_topk_weights: Optional[torch.Tensor],
                                       layout_desc: EPCommLayoutDesc, num_sm: int = 0):
        if num_sm <= 0:
            num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count * 8

        topk = self.ep_context.config.topk
        hidden = self.ep_context.config.hidden
        if dispatch_topk_weights is not None:
            assert dispatch_out.shape[0] == dispatch_topk_weights.shape[0]
        assert dispatch_out.shape[0] == layout_desc.recv_topk_scatter_indices.shape[0]
        assert dispatch_out.shape[1] == hidden
        assert layout_desc.recv_topk_scatter_indices.shape[1] == topk
        if dispatch_topk_weights is not None:
            assert dispatch_topk_weights.shape[1] == topk
        assert layout_desc.recv_token_count is not None

        # postprocess:
        # 1. local dispatch(inplace)
        # 2. copy recv_topk_scatter_indices from symm tensor to torch tensor
        # 3. get dispatch weights
        buffer_size = dispatch_out.shape[0]
        recv_topk_scatter_indices = torch.empty((buffer_size, topk), dtype=layout_desc.recv_topk_scatter_indices.dtype,
                                                device=layout_desc.recv_topk_scatter_indices.device)
        dispatch_weights = None
        if dispatch_topk_weights is not None:
            dispatch_weights = torch.empty((buffer_size, ), dtype=dispatch_topk_weights.dtype,
                                           device=dispatch_topk_weights.device)
        self._buffer_in_range(layout_desc.recv_topk_scatter_indices, self.ep_context.dispatch_topk_scatter_indices_buf)
        if layout_desc.expert_alignment > 1:
            assert layout_desc.recv_aligned_token_count is not None, \
                "recv_aligned_token_count must be set when expert_alignment > 1"
            postprocess_token_count = layout_desc.recv_aligned_token_count
        else:
            postprocess_token_count = layout_desc.recv_token_count
        _ep.dispatch_postprocess(
            dispatch_out,
            layout_desc.recv_topk_scatter_indices,  # comm buffer
            dispatch_topk_weights,
            postprocess_token_count,
            dispatch_weights,
            recv_topk_scatter_indices,  # torch tensor
            hidden,
            topk,
            self.rank,
            self.world_size,
            num_sm,
        )

        # update layout desc
        layout_desc.recv_topk_scatter_indices = recv_topk_scatter_indices
        return dispatch_out, dispatch_weights, layout_desc

    def ep_group_barrier(self):
        if self.ep_context.config.nnodes == 1:
            assert self.ep_context.nvl_barrier_buf.dtype == torch.int32
            _ep.barrier_all_on_stream(self.ep_context.nvl_barrier_buf_ptrs, self.rank, self.world_size)
        else:
            _ep_inter.barrier_all_on_stream()

    def compute_stable_local_token_within_expert_offset_and_expert_counts(self, topk_indices: torch.Tensor, num_sm=-1):
        token_within_expert_offset, block_cumsum_hist, expert_counts = _ep.compute_stable_local_token_within_expert_offset_and_expert_counts(
            topk_indices, self.ep_context.config.num_experts, num_sm)
        return token_within_expert_offset, expert_counts

    def get_combine_buffer(self, num_recv_tokens: int, dtype: torch.dtype = None) -> torch.Tensor:
        max_capacity = self.ep_context.combine_input_buf.shape[0]
        if num_recv_tokens > max_capacity:
            raise ValueError(f"num_recv_tokens ({num_recv_tokens}) exceeds combine_buffer capacity ({max_capacity}). ")

        if num_recv_tokens < 0:
            raise ValueError(f"num_recv_tokens must be positive, got {num_recv_tokens}")

        combine_input_buf = self.ep_context.combine_input_buf[:num_recv_tokens]

        if dtype is not None and dtype != combine_input_buf.dtype:
            raise ValueError(f"dtype mismatch, got {dtype}, expected {combine_input_buf.dtype}")
        return combine_input_buf

    def _buffer_in_range(self, tensor: torch.Tensor, buf: torch.Tensor) -> bool:
        if tensor.numel() == 0:
            return True

        assert tensor.dtype == buf.dtype
        assert tensor.device == buf.device
        assert tensor.is_contiguous()

        tensor_start = tensor.data_ptr()
        tensor_end = tensor_start + tensor.numel() * tensor.element_size()
        buf_start = buf.data_ptr()
        buf_end = buf_start + buf.numel() * buf.element_size()
        return tensor_start >= buf_start and tensor_end <= buf_end

    def _validate_combine_input_buffer(self, tensor: torch.Tensor) -> bool:
        return self._buffer_in_range(tensor, self.ep_context.combine_input_buf)

    def _validate_combine_weight_buffer(self, tensor: torch.Tensor) -> bool:
        return self._buffer_in_range(tensor, self.ep_context.combine_topk_weights_buf)

    def combine_intranode_preprocess(self, input: torch.Tensor, layout_desc: EPCommLayoutDesc,
                                     weight: Optional[torch.Tensor] = None, zero_copy: bool = False, num_sm: int = 0):
        if num_sm <= 0:
            num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count * 8

        assert layout_desc.recv_token_count is not None
        assert layout_desc.recv_topk_scatter_indices is not None
        assert input.shape[0] <= self.ep_context.combine_input_buf.shape[0]
        if zero_copy:
            assert self._validate_combine_input_buffer(input)
            combine_input_buf = input
        else:
            combine_input_buf = self.ep_context.combine_input_buf[:input.shape[0]]
            combine_input_buf.copy_(input)

        if weight is not None:
            assert len(weight.shape) == 1 and weight.shape[0] == input.shape[0]
            assert weight.dtype == self.ep_context.config.weight_dtype
            assert weight.is_contiguous()
            combine_input_weight_buf = self.ep_context.combine_topk_weights_buf[:input.shape[0]]
        else:
            combine_input_weight_buf = None
        if layout_desc.expert_alignment > 1:
            assert layout_desc.recv_aligned_token_count is not None, \
                "recv_aligned_token_count must be set when expert_alignment > 1"
            combine_token_count = layout_desc.recv_aligned_token_count
        else:
            combine_token_count = layout_desc.recv_token_count
        _ep.combine_preprocess_inplace(
            combine_input_buf,
            combine_token_count,
            layout_desc.recv_topk_scatter_indices,
            self.rank,
            self.world_size,
            num_sm,
            weight,
            combine_input_weight_buf,
        )
        return combine_input_buf, combine_input_weight_buf

    def combine_intranode(self, input_preprocessed: torch.Tensor, layout_desc: EPCommLayoutDesc,
                          weight_preprocessed: Optional[torch.Tensor] = None):
        layout_desc.check_combine_required_inputs()
        assert self._validate_combine_input_buffer(input_preprocessed)
        has_weight = weight_preprocessed is not None
        if has_weight:
            assert self._validate_combine_weight_buffer(weight_preprocessed)

        # pull mode, wait ep ranks to finish preprocess
        self.ep_group_barrier()
        hidden = self.ep_context.config.hidden
        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        topk = self.ep_context.config.topk
        combine_intranode_out_buf = torch.empty((layout_desc.token_dst_scatter_indices.shape[0], hidden),
                                                dtype=input_preprocessed.dtype, device=input_preprocessed.device)
        if has_weight:
            combine_intranode_input_weight_ptrs = self.ep_context.combine_topk_weights_buf_ptrs
            combine_intranode_out_weight_buf = torch.empty((layout_desc.token_dst_scatter_indices.shape[0], topk),
                                                           dtype=self.ep_context.config.weight_dtype,
                                                           device=input_preprocessed.device)
        else:
            combine_intranode_input_weight_ptrs = None
            combine_intranode_out_weight_buf = None
        _ep.combine_intranode(
            self.ep_context.combine_input_buf_ptrs,
            layout_desc.token_topk_send_mask,
            layout_desc.topk_indices,
            layout_desc.token_dst_scatter_indices,
            combine_intranode_out_buf,
            self.rank,
            self.world_size,
            num_experts_per_rank,
            self.num_sm,
            combine_intranode_input_weight_ptrs,
            combine_intranode_out_weight_buf,
        )
        self.ep_group_barrier()
        return combine_intranode_out_buf, combine_intranode_out_weight_buf

    def dispatch_internode(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: Optional[torch.Tensor],
                           layout_desc: EPCommLayoutDesc, num_qps: Optional[int] = None):
        max_slot_num_token = self.ep_context.config.max_m
        if layout_desc.need_recompute_token_within_expert_offset_and_expert_counts(topk_indices):
            layout_desc.token_within_expert_offset, layout_desc.expert_counts = \
                self.compute_stable_local_token_within_expert_offset_and_expert_counts(topk_indices, self.num_sm)

        num_token = input.shape[0]
        rdma_rail_send_views = self.ep_context.rdma_rail_send_slot_views(
            num_token,
            self.ep_context.config.hidden,
            self.ep_context.config.topk,
            max_slot_num_token=max_slot_num_token,
        )
        if layout_desc.need_recompute_dispatch_layout(self.expert_alignment, num_token):
            layout_desc.num_tokens_per_rank = torch.empty((self.ep_context.config.world_size, ), dtype=torch.int32,
                                                          device=input.device)
            (
                layout_desc.recv_base_offset,
                layout_desc.token_dst_scatter_indices,
                layout_desc.token_topk_send_mask,
                layout_desc.recv_token_count_cpu,
                layout_desc.recv_token_count,
                layout_desc.recv_aligned_token_count_cpu,
                layout_desc.recv_aligned_token_count,
                layout_desc.recv_expert_counts,
            ) = _ep_inter.compute_dispatch_layout(
                topk_indices,
                layout_desc.token_within_expert_offset,
                layout_desc.expert_counts,
                self.ep_context.full_splits_win_handle,
                layout_desc.num_tokens_per_rank,
                self.ep_context.config.num_experts,
                self.num_sm,
                self.ep_context.recv_token_count_cpu,
                expert_alignment=self.expert_alignment,
                rdma_topk_send_mask=rdma_rail_send_views["topk_send_mask"],
                rdma_token_dst_scatter=rdma_rail_send_views["token_dst_scatter"],
            )
            layout_desc.expert_alignment = self.expert_alignment
            layout_desc.num_tokens = num_token
        else:
            if layout_desc.num_tokens_per_rank is None:
                raise ValueError("layout_desc.num_tokens_per_rank is required for internode dispatch reuse")

        if layout_desc.expert_alignment > 1:
            buf_count_cpu = layout_desc.recv_aligned_token_count_cpu
            buf_count_gpu = layout_desc.recv_aligned_token_count
        else:
            buf_count_cpu = layout_desc.recv_token_count_cpu
            buf_count_gpu = layout_desc.recv_token_count
        dispatch_recv_token_count, _ = self._realloc_dispatch_output_buf(buf_count_cpu, buf_count_gpu)

        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        max_recv_tokens = self.ep_context.dispatch_output_buf.shape[0]
        meta_shape = (self.ep_context.config.nnodes, max_slot_num_token, self.ep_context.config.topk)
        layout_desc.topk_indices = torch.empty(meta_shape, dtype=topk_indices.dtype, device=topk_indices.device)
        layout_desc.token_topk_send_mask = torch.empty(meta_shape, dtype=torch.int32, device=topk_indices.device)
        layout_desc.token_dst_scatter_indices = torch.empty(meta_shape, dtype=torch.int32, device=topk_indices.device)
        rdma_rail_send_views["x"].copy_(input)
        rdma_rail_send_views["topk_indices"].copy_(topk_indices)
        if topk_weights is not None:
            rdma_rail_send_views["topk_weights"].copy_(topk_weights)
        self.ep_group_barrier()
        _ep_inter.dispatch_internode(
            self.ep_context.dispatch_output_buf_ptrs,
            self.ep_context.dispatch_topk_weights_buf_ptrs,
            self.ep_context.dispatch_topk_scatter_indices_buf_ptrs,
            max_recv_tokens,
            self.ep_context.rdma_rail_send_buf,
            self.ep_context.rdma_rail_send_win_handle,
            layout_desc.num_tokens_per_rank,
            layout_desc.topk_indices,
            layout_desc.token_topk_send_mask,
            layout_desc.token_dst_scatter_indices,
            max_slot_num_token,
            num_token,
            self.ep_context.config.hidden,
            topk_weights is not None,
            num_experts_per_rank,
            self.num_sm,
            num_qps,
        )
        # Fused reset+barrier: dispatch signals are quiescent here and must be
        # reset before the next dispatch; plain barriers do not touch signals.
        _ep_inter.reset_signals_barrier_all_on_stream(*self._dispatch_signal_range)
        layout_desc.recv_topk_scatter_indices = self.ep_context.dispatch_topk_scatter_indices_buf[:
                                                                                                  dispatch_recv_token_count]
        dispatch_weights = None
        if topk_weights is not None:
            dispatch_weights = self.ep_context.dispatch_topk_weights_buf[:dispatch_recv_token_count]
        return (self.ep_context.dispatch_output_buf[:dispatch_recv_token_count], dispatch_weights, layout_desc)

    def combine_internode(self, input_preprocessed: torch.Tensor, layout_desc: EPCommLayoutDesc,
                          weight_preprocessed: Optional[torch.Tensor] = None, num_qps: Optional[int] = None):
        layout_desc.check_combine_required_inputs()
        assert self._validate_combine_input_buffer(input_preprocessed)
        has_weight = weight_preprocessed is not None
        if has_weight:
            assert self._validate_combine_weight_buffer(weight_preprocessed)

        self.ep_group_barrier()
        hidden = self.ep_context.config.hidden
        topk = self.ep_context.config.topk
        num_experts_per_rank = self.ep_context.config.num_experts // self.ep_context.config.world_size
        if layout_desc.token_within_expert_offset is None:
            raise ValueError("layout_desc.token_within_expert_offset is required for internode combine")
        num_token = layout_desc.token_within_expert_offset.shape[0]
        combine_out = torch.empty((num_token, hidden), dtype=input_preprocessed.dtype, device=input_preprocessed.device)
        kernel_combine_weight = None
        weight_ptrs = None
        if layout_desc.num_tokens_per_rank is None:
            raise ValueError("layout_desc.num_tokens_per_rank is required for internode combine")
        if (layout_desc.topk_indices is None or layout_desc.token_topk_send_mask is None
                or layout_desc.token_dst_scatter_indices is None):
            raise ValueError("internode combine requires per-node topk_indices/topk_send_mask/token_dst_scatter")
        if has_weight:
            weight_ptrs = self.ep_context.combine_topk_weights_buf_ptrs
            kernel_combine_weight = torch.empty((num_token, topk), dtype=self.ep_context.config.weight_dtype,
                                                device=input_preprocessed.device)
        _ep_inter.combine_internode(
            self.ep_context.combine_input_buf_ptrs,
            weight_ptrs,
            self.ep_context.rdma_rail_send_buf,
            self.ep_context.rdma_rail_send_win_handle,
            layout_desc.num_tokens_per_rank,
            combine_out,
            layout_desc.topk_indices,
            layout_desc.token_topk_send_mask,
            layout_desc.token_dst_scatter_indices,
            kernel_combine_weight,
            self.ep_context.config.max_m,
            num_experts_per_rank,
            self.num_sm,
            num_qps,
        )
        # Fused reset+barrier for the combine signal range (see dispatch).
        _ep_inter.reset_signals_barrier_all_on_stream(*self._combine_signal_range)
        return combine_out, kernel_combine_weight

    def dispatch(self, input: torch.Tensor, topk_indices: torch.Tensor, topk_weights: Optional[torch.Tensor],
                 layout_desc: EPCommLayoutDesc = None, num_qps: Optional[int] = None):
        # Contract: topk_indices must be in [0, num_experts], where num_experts
        # is the drop sentinel.  Negative values are not a supported sentinel.
        if layout_desc is None:
            layout_desc = EPCommLayoutDesc()
        else:
            layout_desc.check_layout_desc(num_tokens=topk_indices.shape[0], topk=topk_indices.shape[1],
                                          num_experts=self.ep_context.config.num_experts,
                                          world_size=self.ep_context.config.world_size,
                                          local_world_size=self.ep_context.config.local_world_size,
                                          max_slot_num_token=self.ep_context.config.max_m)

        if self.ep_context.config.nnodes == 1:
            # num_qps only affects internode RDMA traffic; ignored intranode so
            # the same caller code runs on a single node.
            return self.dispatch_intranode(input, topk_indices, topk_weights, layout_desc)
        else:
            return self.dispatch_internode(input, topk_indices, topk_weights, layout_desc, num_qps=num_qps)

    def dispatch_postprocess(self, dispatch_out: torch.Tensor, dispatch_topk_weights: Optional[torch.Tensor],
                             layout_desc: EPCommLayoutDesc, num_sm: int = 0):
        return self.dispatch_intranode_postprocess(dispatch_out, dispatch_topk_weights, layout_desc, num_sm)

    def combine_preprocess(self, input: torch.Tensor, layout_desc: EPCommLayoutDesc,
                           weight: Optional[torch.Tensor] = None, zero_copy: bool = False, num_sm: int = 0):
        return self.combine_intranode_preprocess(input, layout_desc, weight=weight, zero_copy=zero_copy, num_sm=num_sm)

    def combine(self, input_preprocessed: torch.Tensor, layout_desc: EPCommLayoutDesc,
                weight_preprocessed: Optional[torch.Tensor] = None, num_qps: Optional[int] = None):
        if self.ep_context.config.nnodes == 1:
            # num_qps ignored intranode (see dispatch).
            return self.combine_intranode(input_preprocessed, layout_desc=layout_desc,
                                          weight_preprocessed=weight_preprocessed)
        else:
            return self.combine_internode(input_preprocessed, layout_desc=layout_desc,
                                          weight_preprocessed=weight_preprocessed, num_qps=num_qps)
