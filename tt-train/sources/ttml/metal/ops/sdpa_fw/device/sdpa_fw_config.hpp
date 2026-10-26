// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "metal/common/const_utils.hpp"
#include "metal/ops/sdpa_fw/sdpa_fw_program_config.hpp"

// Everything the SDPA forward program depends on besides the kernels themselves: how the user knobs
// turn into chunk sizes, how much L1 the circular buffers take, how the work items are shared out and
// the cost model that picks the default. Shared by validate (early errors) and the program factory
// (which also serves the ring SDPA path).
namespace ttml::metal::ops::sdpa_fw::device {

struct SDPAForwardShape {
    uint32_t batch;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t St;   // sequence tiles
    uint32_t qWt;  // Q/K inner-dim tiles
    uint32_t vWt;  // V inner-dim tiles
    AttentionMaskType mask_type;
    bool has_gate;
    bool return_intermediates;
};

// The program config in tiles. rows = Sq_chunk_t * heads_per_item tile-rows make one work item.
struct SDPAForwardChunks {
    uint32_t Sq_chunk_t;
    uint32_t Sk_chunk_t;
    uint32_t heads_per_item;

    uint32_t rows() const {
        return Sq_chunk_t * heads_per_item;
    }
};

// Circular-buffer depths in tiles; 0 = the CB is not created.
struct SDPAForwardCbTiles {
    uint32_t query;
    uint32_t key;
    uint32_t value;
    uint32_t attn_mask;
    uint32_t gate;
    uint32_t intermediate;
    uint32_t attention_weights;
    uint32_t stat;    // each of max prev/cur, exp max diff, sum prev/cur
    uint32_t mm_out;  // each of prev/cur
    uint32_t output;
};

SDPAForwardCbTiles cb_tiles(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks);
uint64_t l1_bytes(const SDPAForwardCbTiles& tiles);

// Work item = one Q chunk of one head stack, see kernels/sdpa_fw_work_list.hpp. Each core gets a list
// of units: a run of whole items, or one K/V step range of one item that is sent to / merged on a core.
enum class UnitRole : uint32_t { Whole = 0, Sender = 1, Merger = 2 };

struct WorkUnit {
    uint32_t item = 0;
    UnitRole role = UnitRole::Whole;
    uint32_t num_items = 1;   // Whole: run length
    uint32_t step_begin = 0;  // Sender / Merger
    uint32_t step_end = 0;
    uint32_t num_partials = 0;  // Merger
    uint32_t sem_idx = 0;       // Merger: semaphore on its core the senders bump
    uint32_t slot = 0;          // Sender: scratch slot; Merger: first of its partials' slots
    uint32_t merger_core = 0;   // Sender: logical core index of its merger
};
using CoreUnits = std::vector<WorkUnit>;
constexpr uint32_t kWordsPerUnit = 3U;

// Runtime-arg words of one unit; merger_noc_x/y = the merger core's NoC coordinates (senders only).
void append_unit_words(
    std::vector<uint32_t>& words, const WorkUnit& unit, uint32_t merger_noc_x, uint32_t merger_noc_y);

uint32_t num_work_items(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks);
uint32_t num_kv_steps(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, uint32_t item);

// Per-core units for num_cores cores: causal items longest first, otherwise contiguous blocks; falls
// back to contiguous blocks when the lists would not fit the runtime args.
std::vector<CoreUnits> split_work(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, uint32_t num_cores);

// Predicted time of the busiest core in microseconds (Blackhole constants; the ranking is what matters).
double predict_us(
    const SDPAForwardShape& shape,
    const SDPAForwardChunks& chunks,
    const std::vector<CoreUnits>& per_core_units,
    uint32_t num_cores);

// Cheap knob checks for validate: each set knob must divide its axis, and a fully specified config must
// fit l1_budget_bytes. Throws otherwise. (Picking the unset knobs is a search; the program factory does
// that once per program-cache miss with resolve_chunks.)
void check_chunks(const SDPAForwardProgramConfig& config, const SDPAForwardShape& shape, uint64_t l1_budget_bytes);

// check_chunks, then picks the unset knobs by the cost model among the configs that fit l1_budget_bytes.
// Throws when nothing fits.
SDPAForwardChunks resolve_chunks(
    const SDPAForwardProgramConfig& config,
    const SDPAForwardShape& shape,
    uint64_t l1_budget_bytes,
    uint32_t num_available_cores);

}  // namespace ttml::metal::ops::sdpa_fw::device
