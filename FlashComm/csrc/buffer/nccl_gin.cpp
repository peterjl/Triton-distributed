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

#include "flash_comm/buffer/nccl_gin.h"
#include "flash_comm/common.h"
#include "flash_comm/ep/internode.h"
#include "flash_comm/nccl_utils.h"

#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace flash_comm {
namespace buffer {

namespace {

std::mutex g_mutex;
NcclGinState g_state;

void destroy_unlocked() {
  auto &st = g_state;
  if (!st.initialized && st.comm == nullptr) {
    return;
  }
  if (st.comm != nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
    if (st.initialized) {
      NCCL_CHECK(ncclDevCommDestroy(st.comm, &st.dev_comm));
    }
    NCCL_CHECK(ncclCommFinalize(st.comm));
    ncclResult_t async_result = ncclInProgress;
    while (async_result == ncclInProgress) {
      NCCL_CHECK(ncclCommGetAsyncError(st.comm, &async_result));
      if (async_result == ncclInProgress) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    NCCL_CHECK(async_result);
    NCCL_CHECK(ncclCommDestroy(st.comm));
  }
  st = NcclGinState{};
}

void validate_init_args(int rank, int nranks, int local_world_size,
                        int gin_contexts, int gin_signals, int rail_barriers,
                        int gin_queue_depth, int gin_connection_type,
                        int ep_num_qps) {
  if (local_world_size <= 0 || nranks % local_world_size != 0) {
    throw std::runtime_error("nranks must be divisible by local_world_size");
  }
  if (rank < 0 || rank >= nranks) {
    throw std::runtime_error("invalid rank");
  }
  if (gin_contexts <= 0 || gin_signals <= 0 || rail_barriers <= 0 ||
      gin_queue_depth < 0) {
    throw std::runtime_error("invalid NCCL GIN resource configuration");
  }
  if (ep_num_qps <= 0 || ep_num_qps > gin_contexts) {
    throw std::runtime_error("ep_num_qps must be in [1, gin_contexts]");
  }
  const int nnodes = nranks / local_world_size;
  const int required_ep_signals =
      ::flash_comm::ep::internode::ep_required_gin_signal_count(nnodes);
  if (gin_signals < required_ep_signals) {
    throw std::runtime_error("gin_signals is too small for per-chunk dispatch "
                             "and per-QP combine signals");
  }
  if (gin_connection_type < static_cast<int>(NCCL_GIN_CONNECTION_NONE) ||
      gin_connection_type > static_cast<int>(NCCL_GIN_CONNECTION_RAIL)) {
    throw std::runtime_error("invalid NCCL GIN connection type");
  }
}

void fill_common_state(NcclGinState &st, int rank, int nranks,
                       int local_world_size, int gin_contexts, int ep_num_qps) {
  st.rank = rank;
  st.nranks = nranks;
  st.local_world_size = local_world_size;
  st.nnodes = nranks / local_world_size;
  st.local_rank = rank % local_world_size;
  st.gin_contexts = gin_contexts;
  st.ep_num_qps = ep_num_qps;
}

void init_comm(NcclGinState &st, const ncclUniqueId &nccl_id, int rank,
               int nranks, int local_world_size, int gin_contexts,
               int gin_signals, int rail_barriers, int gin_queue_depth,
               int gin_connection_type) {
  NCCL_CHECK(ncclCommInitRank(&st.comm, nranks, nccl_id, rank));
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(st.comm, &props));
  st.gin_type = static_cast<int>(props.ginType);
  if (!props.deviceApiSupport || props.ginType == NCCL_GIN_TYPE_NONE) {
    throw std::runtime_error(
        "NCCL GIN unavailable: set NCCL_GIN_ENABLE=1 NCCL_GIN_TYPE=3");
  }

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = false;
  reqs.lsaBarrierCount = rail_barriers;
  reqs.railGinBarrierCount = rail_barriers;
  reqs.ginSignalCount = gin_signals;
  reqs.ginContextCount = gin_contexts;
  reqs.ginQueueDepth = gin_queue_depth;
  reqs.ginConnectionType =
      static_cast<ncclGinConnectionType_t>(gin_connection_type);
  NCCL_CHECK(ncclDevCommCreate(st.comm, &reqs, &st.dev_comm));
  st.initialized = true;
  st.lsa_rank = st.dev_comm.lsaRank;
  st.lsa_size = st.dev_comm.lsaSize;
  if (st.lsa_size < local_world_size ||
      (st.lsa_rank % local_world_size) != (rank % local_world_size)) {
    throw std::runtime_error(
        "NCCL GIN LSA team does not cover local_world_size/local_rank");
  }
}

} // namespace

int nccl_gin_unique_id_bytes() {
  return static_cast<int>(sizeof(ncclUniqueId));
}

void nccl_gin_get_unique_id(void *uid_out) {
  if (uid_out == nullptr) {
    throw std::runtime_error("uid_out is null");
  }
  ncclUniqueId id{};
  NCCL_CHECK(ncclGetUniqueId(&id));
  std::memcpy(uid_out, &id, sizeof(ncclUniqueId));
}

int nccl_gin_init_rank(const void *uid, int uid_len, int rank, int nranks,
                       int local_world_size, int gin_contexts, int gin_signals,
                       int rail_barriers, int gin_queue_depth,
                       int gin_connection_type, int ep_num_qps) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_state.initialized) {
    throw std::runtime_error("nccl_gin_init_rank: already initialized");
  }
  if (uid == nullptr || uid_len != static_cast<int>(sizeof(ncclUniqueId))) {
    throw std::runtime_error("invalid NCCL unique id length");
  }
  validate_init_args(rank, nranks, local_world_size, gin_contexts, gin_signals,
                     rail_barriers, gin_queue_depth, gin_connection_type,
                     ep_num_qps);

  ncclUniqueId nccl_id{};
  std::memcpy(&nccl_id, uid, sizeof(ncclUniqueId));

  auto &st = g_state;
  fill_common_state(st, rank, nranks, local_world_size, gin_contexts,
                    ep_num_qps);

  try {
    init_comm(st, nccl_id, rank, nranks, local_world_size, gin_contexts,
              gin_signals, rail_barriers, gin_queue_depth, gin_connection_type);
    return st.gin_type;
  } catch (...) {
    try {
      destroy_unlocked();
    } catch (...) {
    }
    throw;
  }
}

void nccl_gin_destroy_rank() {
  std::lock_guard<std::mutex> lock(g_mutex);
  destroy_unlocked();
}

int nccl_gin_is_initialized() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_state.initialized ? 1 : 0;
}

NcclGinState &nccl_gin_require_state() {
  if (!g_state.initialized) {
    throw std::runtime_error("NCCL GIN is not initialized");
  }
  return g_state;
}

ncclComm_t nccl_gin_comm() { return nccl_gin_require_state().comm; }

const ncclDevComm *nccl_gin_dev_comm() {
  return &nccl_gin_require_state().dev_comm;
}

int nccl_gin_lsa_rank() { return nccl_gin_require_state().lsa_rank; }

int nccl_gin_lsa_size() { return nccl_gin_require_state().lsa_size; }

} // namespace buffer
} // namespace flash_comm
