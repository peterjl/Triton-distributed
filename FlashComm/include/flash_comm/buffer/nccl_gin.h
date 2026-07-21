/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

namespace flash_comm {
namespace buffer {

struct NcclGinState {
  int rank = -1;
  int nranks = 0;
  int local_rank = 0;
  int local_world_size = 0;
  int lsa_rank = 0;
  int lsa_size = 0;
  int nnodes = 0;
  int node_id = 0;
  ncclComm_t comm = nullptr;
  ncclDevComm dev_comm{};
  int gin_type = 0;
  int gin_contexts = 0;
  int gin_signals = 0;
  int ep_num_qps = 1;
  uint64_t dispatch_signal_epoch = 0;
  uint64_t combine_signal_epoch = 0;
  bool initialized = false;
};

int nccl_gin_unique_id_bytes();
void nccl_gin_get_unique_id(void *uid_out);

int nccl_gin_init_rank(const void *uid, int uid_len, int rank, int nranks,
                       int local_world_size, int gin_contexts, int gin_signals,
                       int rail_barriers, int gin_queue_depth,
                       int gin_connection_type, int ep_num_qps = 1);

void nccl_gin_destroy_rank();
int nccl_gin_is_initialized();

NcclGinState &nccl_gin_require_state();
ncclComm_t nccl_gin_comm();
const ncclDevComm *nccl_gin_dev_comm();
int nccl_gin_lsa_rank();
int nccl_gin_lsa_size();
uint64_t nccl_gin_next_dispatch_signal_epoch(uint64_t signal_increments = 1);
uint64_t nccl_gin_next_combine_signal_epoch(uint64_t signal_increments = 1);

} // namespace buffer
} // namespace flash_comm
