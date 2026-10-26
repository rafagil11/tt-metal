// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Work item = Sq_chunk_t query tile-rows of heads_per_item consecutive Q heads (one K/V group),
// item = (batch * stacks_per_batch + stack) * q_num_chunks + chunk; row r of an item is head
// head0 + r / Sq_chunk_t, tile row first_row_tile + r % Sq_chunk_t.
//
// A core's work list is a sequence of units, three runtime-arg words each (see sdpa_fw_config.cpp):
//   whole:  a run of consecutive items, each over its full K/V range;
//   sender: K/V steps [step_begin, step_end) of one item, result sent to a merger core as a partial;
//   merger: its own step range of one item, then the senders' partials folded in, then finalized.
// All three kernels walk the same list.
constexpr uint32_t kUnitWhole = 0U;
constexpr uint32_t kUnitSender = 1U;
constexpr uint32_t kUnitMerger = 2U;

struct WorkUnit {
    uint32_t item;
    uint32_t role;
    uint32_t num_items;     // whole
    uint32_t step_begin;    // sender / merger
    uint32_t step_end;      // sender / merger
    uint32_t num_partials;  // merger
    uint32_t sem_idx;       // merger: which of its core's semaphores the senders bump
    uint32_t slot;          // sender: scratch slot it writes; merger: first slot it reads
    uint32_t merger_x;      // sender: merger core, NoC coordinates
    uint32_t merger_y;
};

template <typename Fn>
inline void for_each_work_unit(uint32_t arg_idx, Fn&& fn) {
    const uint32_t num_units = get_arg_val<uint32_t>(arg_idx++);
    for (uint32_t u = 0; u < num_units; ++u) {
        const uint32_t w0 = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t w1 = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t w2 = get_arg_val<uint32_t>(arg_idx++);
        WorkUnit unit{};
        unit.item = w0 & 0xFFFFFU;
        unit.role = (w0 >> 20U) & 0x3U;
        unit.num_partials = (w0 >> 22U) & 0x3FU;
        unit.sem_idx = (w0 >> 28U) & 0xFU;
        if (unit.role == kUnitWhole) {
            unit.num_items = w1;
        } else {
            unit.num_items = 1U;
            unit.step_begin = w1 & 0xFFFFU;
            unit.step_end = w1 >> 16U;
        }
        unit.slot = w2 & 0xFFFFU;
        unit.merger_x = (w2 >> 16U) & 0xFFU;
        unit.merger_y = (w2 >> 24U) & 0xFFU;
        fn(unit);
    }
}

struct WorkItemCoords {
    uint32_t batch;
    uint32_t head0;
    uint32_t first_row_tile;
};

template <uint32_t q_num_chunks, uint32_t Sq_chunk_t, uint32_t q_heads, uint32_t heads_per_item>
inline WorkItemCoords decode_work_item(uint32_t item) {
    constexpr uint32_t stacks_per_batch = q_heads / heads_per_item;
    const uint32_t stack = item / q_num_chunks;
    return {stack / stacks_per_batch, (stack % stacks_per_batch) * heads_per_item, (item % q_num_chunks) * Sq_chunk_t};
}

// K/V steps of Sk_chunk_t tiles a chunk attends to: causal up to its last diagonal tile, else all.
template <uint32_t Ht, uint32_t Sq_chunk_t, uint32_t Sk_chunk_t>
inline uint32_t num_kv_steps_of_chunk([[maybe_unused]] uint32_t first_row_tile) {
#ifdef CAUSAL_MASK
    return (first_row_tile + Sq_chunk_t + Sk_chunk_t - 1U) / Sk_chunk_t;
#else
    return Ht / Sk_chunk_t;
#endif
}
