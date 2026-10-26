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
    uint32_t partial;  // each of partial in / partial out (split units)
    uint32_t alpha;    // the partial's rescale factor (split units)
};

// with_partials: the CBs of split units (sender / merger) are present.
SDPAForwardCbTiles cb_tiles(const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, bool with_partials);
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

constexpr uint32_t kMaxPieces = 63U;         // num_partials has 6 bits in the unit word
constexpr uint32_t kMaxMergersPerCore = 4U;  // semaphores per core
constexpr uint32_t kDefaultMaxPieces = 4U;

uint32_t partial_tiles_per_row(const SDPAForwardShape& shape);  // a sender's row: max, sum, output tiles

struct SDPAForwardSchedule {
    std::vector<CoreUnits> per_core;
    uint32_t num_slots = 0;  // scratch slots (one per sender)
};

// Work for num_cores cores. Causal: items heavier than a core's fair share are cut into up to
// max_pieces step ranges (senders + one merger), then all pieces go longest first to the least loaded
// core; senders run before mergers on every core, so a merger never waits for a piece behind another
// merger. Everything else: contiguous blocks. Falls back to fewer pieces, then contiguous blocks, when the
// lists would not fit the runtime args.
SDPAForwardSchedule make_schedule(
    const SDPAForwardShape& shape, const SDPAForwardChunks& chunks, uint32_t num_cores, uint32_t max_pieces);

// Predicted time of the busiest core in microseconds (Blackhole constants; the ranking is what matters).
double predict_us(
    const SDPAForwardShape& shape,
    const SDPAForwardChunks& chunks,
    const SDPAForwardSchedule& schedule,
    uint32_t num_cores);

uint32_t effective_max_pieces(const SDPAForwardProgramConfig& config);
bool may_split(const SDPAForwardProgramConfig& config, const SDPAForwardShape& shape);
uint32_t effective_num_cores(const SDPAForwardProgramConfig& config, uint32_t grid_cores);

// Cheap knob checks for validate: each set knob must divide its axis, and a fully specified config must
// fit l1_budget_bytes. Throws otherwise. (Picking the unset knobs is a search; the program factory does
// that once per program-cache miss with resolve_chunks.)
void check_chunks(
    const SDPAForwardProgramConfig& config, const SDPAForwardShape& shape, uint64_t l1_budget_bytes, uint32_t grid_cores);

// check_chunks, then picks the unset chunk knobs by the cost model (with the config's pieces and cores)
// among the configs that fit l1_budget_bytes. Throws when nothing fits.
SDPAForwardChunks resolve_chunks(
    const SDPAForwardProgramConfig& config,
    const SDPAForwardShape& shape,
    uint64_t l1_budget_bytes,
    uint32_t grid_cores);

}  // namespace ttml::metal::ops::sdpa_fw::device
