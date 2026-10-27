// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "llk_defs.h"
#include "llk_math_common.h"
#include "tensor_shape.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

// The reduce unpack delivers the whole data tile with one dvalid. Each face sits in its own
// MAX_FACE_R_DIM-row slot of the source register; faces shorter than that are padded with the pool identity.
constexpr static std::uint32_t REDUCE_SRC_FACE_ROWS = MAX_FACE_R_DIM;

// Dest row where REDUCE_SCALAR pools all faces before the transpose. Must be past row 0, where the result goes,
// and inside the smallest dest tile (16 rows).
constexpr static std::uint32_t REDUCE_SCALAR_SCRATCH_DST_ROW = 8;

/**
 * @brief Emit the pool instruction (GMPOOL for MAX, GAPOOL for SUM/AVG) matching the reduce pool type.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @param addr_mod: Address-mod slot applied after the instruction
 * @param dst_row: Dest row offset the pooled row is written to
 */
template <PoolType POOL_TYPE>
inline void _reduce_pool_(const std::uint32_t addr_mod, const std::uint32_t dst_row)
{
    if constexpr (POOL_TYPE == PoolType::MAX)
    {
        TT_GMPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, addr_mod, p_gpool::INDEX_DIS, dst_row);
    }
    else
    {
        TT_GAPOOL(p_gpool::CLR_NONE, p_gpool::DIM_16X16, addr_mod, p_gpool::INDEX_DIS, dst_row);
    }
}

/**
 * @brief Dest rows one face occupies; faces shorter than MAX_FPU_ROWS are padded to MAX_FPU_ROWS rows.
 *
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
constexpr std::uint32_t _reduce_dest_face_rows_(const TensorShape& tensor_shape)
{
    return std::max<std::uint32_t>(tensor_shape.face_r_dim, MAX_FPU_ROWS);
}

/**
 * @brief FPU row groups (ELTWISE_MATH_ROWS each) needed to cover one face; tiny faces need a single group.
 *
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
constexpr std::uint32_t _reduce_row_groups_per_face_(const TensorShape& tensor_shape)
{
    return std::max<std::uint32_t>(tensor_shape.face_r_dim / ELTWISE_MATH_ROWS, 1);
}

//----------------------
// REDUCE_ROW SUM/AVG
//----------------------

/**
 * @brief Sets up addrmods for REDUCE_ROW SUM/AVG.
 *
 * @param fidelity: Resolved source-format phase schedule.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_math_reduce_row_sum_addrmod_(const MathFidelitySchedule fidelity, const TensorShape& tensor_shape)
{
    const std::uint32_t row_groups = _reduce_row_groups_per_face_(tensor_shape);

    // Next row group of the same face
    addr_mod_t {.srcb = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_0);

    // First row group of the next face slot
    addr_mod_t {.srcb = {.incr = static_cast<std::uint8_t>(REDUCE_SRC_FACE_ROWS - (row_groups - 1) * ELTWISE_MATH_ROWS)}}.set(ADDR_MOD_1);

    // End of the tile: rewind the sources for the next fidelity phase
    addr_mod_t {.srca = {.clr = 1}, .srcb = {.clr = 1}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_2);
}

/**
 * @brief Sets up mop config for REDUCE_ROW SUM/AVG.
 *
 * The operands are swapped (see @ref reduce_swaps_operands): SrcA holds the transposed scaler face, so its column 0
 * carries the scaler row, and SrcB holds the data tile. One MVMUL per FPU row group multiplies the data rows by the
 * scaler, leaving each row sum in column 0 of dest. For a 32 x 32 tile:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * Faces 0 and 1 accumulate into dest face 0, faces 2 and 3 into dest face 2. Every fidelity phase replays the whole tile.
 *
 * @param fidelity: Resolved source-format phase schedule.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_math_reduce_row_sum_mop_config_(const MathFidelitySchedule fidelity, const TensorShape& tensor_shape)
{
    const std::uint32_t row_groups     = _reduce_row_groups_per_face_(tensor_shape);
    const std::uint32_t dest_row_pitch = _reduce_dest_face_rows_(tensor_shape) * tensor_shape.num_faces_c_dim;
    const std::uint32_t replay_buf_len = tensor_shape.total_num_faces() * row_groups;

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [&tensor_shape, row_groups, dest_row_pitch, replay_buf_len]
        {
            std::uint32_t instr_idx = 0;
            for (std::uint32_t face_r = 0; face_r < tensor_shape.num_faces_r_dim; face_r++)
            {
                for (std::uint32_t face_c = 0; face_c < tensor_shape.num_faces_c_dim; face_c++)
                {
                    for (std::uint32_t row_group = 0; row_group < row_groups; row_group++)
                    {
                        instr_idx++;
                        const std::uint32_t addr_mod = (instr_idx == replay_buf_len) ? ADDR_MOD_2 : (row_group == row_groups - 1) ? ADDR_MOD_1 : ADDR_MOD_0;
                        TT_MVMUL(p_setrwc::CLR_NONE, 0, addr_mod, face_r * dest_row_pitch + row_group * ELTWISE_MATH_ROWS);
                    }
                }
            }
        });

    ckernel_template temp(1 /*outer loop*/, fidelity.phase_count, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
    temp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, p_setrwc::SET_ABD_F));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

//----------------------
// REDUCE_ROW MAX
//----------------------

/**
 * @brief Sets up addrmods for REDUCE_ROW MAX.
 */
inline void _llk_math_reduce_row_max_addrmod_()
{
    // Next face slot of SrcA
    addr_mod_t {.srca = {.incr = REDUCE_SRC_FACE_ROWS}}.set(ADDR_MOD_0);

    // Last face: rewind SrcA, the transpose adds the zeroed SrcA from row 0
    addr_mod_t {.srca = {.clr = 1}}.set(ADDR_MOD_1);

    // Next row group of the transposed face
    addr_mod_t {.srcb = {.incr = ELTWISE_MATH_ROWS}, .dest = {.incr = ELTWISE_MATH_ROWS}}.set(ADDR_MOD_2);

    // Last row group of the transposed face: rewind for the next face row
    addr_mod_t {.srcb = {.clr = 1}, .dest = {.clr = 1}}.set(ADDR_MOD_3);

    addr_mod_t {}.set(ADDR_MOD_4);
}

/**
 * @brief Sets up mop config for REDUCE_ROW MAX.
 *
 * The unpacker transposes every face of the data tile in SrcA. GMPOOL pools each row of faces into one row of dest,
 * then each pooled row is transposed through SrcB and added to a zeroed SrcA so that it lands in column 0 of dest:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * Faces 0 and 1 pool into dest face 0, faces 2 and 3 into dest face 2. All faces pool before SrcA is zeroed.
 *
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
inline void _llk_math_reduce_row_max_mop_config_(const TensorShape& tensor_shape)
{
    const std::uint32_t row_groups     = _reduce_row_groups_per_face_(tensor_shape);
    const std::uint32_t dest_row_pitch = _reduce_dest_face_rows_(tensor_shape) * tensor_shape.num_faces_c_dim;
    const std::uint32_t num_faces      = tensor_shape.total_num_faces();
    const std::uint32_t replay_buf_len = num_faces + 1 + tensor_shape.num_faces_r_dim * (2 + row_groups);

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [&tensor_shape, row_groups, dest_row_pitch, num_faces]
        {
            for (std::uint32_t face = 0; face < num_faces; face++)
            {
                const std::uint32_t face_r = face / tensor_shape.num_faces_c_dim;
                _reduce_pool_<PoolType::MAX>((face == num_faces - 1) ? ADDR_MOD_1 : ADDR_MOD_0, face_r * dest_row_pitch);
            }

            TTI_ZEROSRC(0, 0, 0, 0, p_zerosrc::READ_BANK, p_zerosrc::CURR_BANK, p_zerosrc::CLR_A);

            for (std::uint32_t face_r = 0; face_r < tensor_shape.num_faces_r_dim; face_r++)
            {
                const std::uint32_t dst_row = face_r * dest_row_pitch;

                // Src B can only transpose rows [16-31], and output them at [32-47]
                TT_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_4, p_movd2b::MOV_1_ROW, 1 /*transpose*/, dst_row);

                // Keep the whole pooled row in dest row 0 of the face, so that accumulating tiles keeps pooling rows
                TT_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_4, p_movd2b::MOV_1_ROW, 0 /*transpose*/, dst_row);

                // Add the transposed rows in SrcB [32-47] to the zeroed SrcA, ELTWISE_MATH_ROWS rows per ELWADDDI.
                // ELWADDDI addresses are in units of 4 rows.
                for (std::uint32_t row_group = 0; row_group < row_groups; row_group++)
                {
                    TT_ELWADDDI(
                        p_elwise::CLR_NONE, 0x0, p_movd2b::SRC_ROW32_OFFSET >> 2, 0x0, (row_group == row_groups - 1) ? ADDR_MOD_3 : ADDR_MOD_2, dst_row >> 2);
                }
            }
        });

    ckernel_template temp(1 /*outer loop*/, 1 /*inner loop*/, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
    temp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, p_setrwc::SET_ABD_F));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

//----------------------
// REDUCE_COL
//----------------------

/**
 * @brief Sets up addrmods for REDUCE_COL.
 *
 * @param fidelity: Resolved source-format phase schedule.
 */
inline void _llk_math_reduce_col_addrmod_(const MathFidelitySchedule fidelity)
{
    // Next face slot of SrcA
    addr_mod_t {.srca = {.incr = REDUCE_SRC_FACE_ROWS}}.set(ADDR_MOD_0);

    // End of the tile: rewind SrcA for the next fidelity phase
    addr_mod_t {.srca = {.clr = 1}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_1);
}

/**
 * @brief Sets up mop config for REDUCE_COL.
 *
 * For reduce col, in a 32 x 32 tile, faces layout would be the following:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * In order to get 1x32 row output (which means 2 output faces, 1x16 row each), Face 0 + Face 2 pool into dest face 0,
 * and Face 1 + Face 3 pool into dest face 1. Every fidelity phase replays the whole tile.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @param fidelity: Resolved source-format phase schedule.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE>
inline void _llk_math_reduce_col_mop_config_(const MathFidelitySchedule fidelity, const TensorShape& tensor_shape)
{
    const std::uint32_t dest_face_rows = _reduce_dest_face_rows_(tensor_shape);
    const std::uint32_t num_faces      = tensor_shape.total_num_faces();

    load_replay_buf(
        0,
        num_faces,
        false,
        0,
        0,
        [&tensor_shape, dest_face_rows, num_faces]
        {
            for (std::uint32_t face = 0; face < num_faces; face++)
            {
                const std::uint32_t face_c = face % tensor_shape.num_faces_c_dim;
                _reduce_pool_<POOL_TYPE>((face == num_faces - 1) ? ADDR_MOD_1 : ADDR_MOD_0, face_c * dest_face_rows);
            }
        });

    ckernel_template temp(1 /*outer loop*/, fidelity.phase_count, TT_OP_REPLAY(0, num_faces, 0, 0, 0, 0));
    temp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, p_setrwc::SET_ABD_F));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

//----------------------
// REDUCE_SCALAR
//----------------------

/**
 * @brief Sets up addrmods for REDUCE_SCALAR.
 *
 * @param fidelity: Phase schedule for the original source operands.
 * @param final_fidelity: Phase schedule for the final pool, whose SrcA is copied back from dest.
 */
inline void _llk_math_reduce_scalar_addrmod_(const MathFidelitySchedule fidelity, const MathFidelitySchedule final_fidelity)
{
    // Next face slot of SrcA
    addr_mod_t {.srca = {.incr = REDUCE_SRC_FACE_ROWS}}.set(ADDR_MOD_0);

    // End of the tile: rewind SrcA for the next fidelity phase
    addr_mod_t {.srca = {.clr = 1}, .fidelity = {.incr = fidelity.phase_increment}}.set(ADDR_MOD_1);

    // Transpose and last final pool: restart the fidelity phases
    addr_mod_t {.fidelity = {.clr = 1}}.set(ADDR_MOD_2);

    // Next fidelity phase of the final pool
    addr_mod_t {.fidelity = {.incr = final_fidelity.phase_increment}}.set(ADDR_MOD_3);
}

/**
 * @brief Sets up mop config for REDUCE_SCALAR.
 *
 * For reduce scalar, in a 32 x 32 tile, faces layout would be the following:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * All faces pool into one scratch row of dest, which is transposed through SrcB into SrcA and pooled again;
 * the result is a single datum placed in datum 0 of the tile.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam EN_32BIT_DEST: Use the active destination addressing mode when clearing the scratch row.
 * @param fidelity: Phase schedule for the original source operands.
 * @param final_fidelity: Phase schedule for the final pool, whose SrcA is copied back from dest.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 */
template <PoolType POOL_TYPE, bool EN_32BIT_DEST>
inline void _llk_math_reduce_scalar_mop_config_(const MathFidelitySchedule fidelity, const MathFidelitySchedule final_fidelity, const TensorShape& tensor_shape)
{
    constexpr std::uint32_t b2a_moves  = FACE_R_DIM / ELTWISE_MATH_ROWS;
    const std::uint32_t num_faces      = tensor_shape.total_num_faces();
    const std::uint32_t final_phases   = final_fidelity.phase_count;
    const std::uint32_t tail_len       = 1 + b2a_moves + 1 + final_phases + 1;
    const std::uint32_t replay_buf_len = num_faces + tail_len;

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [num_faces, final_phases]
        {
            // [0, num_faces): pool all faces into the scratch row, once per fidelity phase
            for (std::uint32_t face = 0; face < num_faces; face++)
            {
                _reduce_pool_<POOL_TYPE>((face == num_faces - 1) ? ADDR_MOD_1 : ADDR_MOD_0, REDUCE_SCALAR_SCRATCH_DST_ROW);
            }

            // [num_faces, num_faces + tail_len): runs once after all fidelity phases
            // Only SrcB transpose can be done from Math
            // Rows [16:31] of SrcB are transposed, then written back into rows [32:47] of SrcB
            // Following will move 1x16 pool result to SrcB to be transposed into 16 rows
            TTI_MOVD2B(0, p_movd2b::SRC_ROW32_OFFSET, ADDR_MOD_2, p_movd2b::MOV_1_ROW, 1 /*transpose*/, REDUCE_SCALAR_SCRATCH_DST_ROW);

#pragma GCC unroll 4
            for (const auto row : fpu_row_offsets<FACE_R_DIM>())
            {
                TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + row, ADDR_MOD_2, FPU_MOV_ROWS, p_movb2a::SRCB_ROW32_OFFSET + row);
            }

            // zero out scratch in dest
            TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, EN_32BIT_DEST, 0, ADDR_MOD_2, REDUCE_SCALAR_SCRATCH_DST_ROW);

            // Do final pool of the transposed rows to generate a single pool datum
            for (std::uint32_t phase = 0; phase < final_phases - 1; phase++)
            {
                _reduce_pool_<POOL_TYPE>(ADDR_MOD_3, 0);
            }
            _reduce_pool_<POOL_TYPE>(ADDR_MOD_2, 0);

            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, p_setrwc::SET_ABD_F);
        });

    ckernel_template temp(1 /*outer loop*/, fidelity.phase_count, TT_OP_REPLAY(0, num_faces, 0, 0, 0, 0));
    temp.set_end_op(TT_OP_REPLAY(num_faces, tail_len, 0, 0, 0, 0));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods and mop config (initialization) for reduce operations.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Sets the reduce dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam EN_32BIT_DEST: Set to true when destination registers use 32-bit addressing.
 * @tparam MATH_FIDELITY_TYPE: Requested multiplication fidelity for SUM/AVG; MAX always runs LoFi, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param src_a_format: Effective SrcA register format.
 * @param src_b_format: Effective SrcB register format.
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc
 * @note On the unpack thread, pair with @ref _llk_unpack_reduce_init_ (T0); on the pack thread, pair with @ref _llk_pack_reduce_mask_config_ (T2).
 * @note @ref _llk_math_reduce_ runs the configured reduction with matching template args.
 * @note Pass the source register formats: for SUM/AVG, init lowers the fidelity to the highest one they can use and skips phases they
 *       cannot use. Run init again whenever either format changes.
 * @note REDUCE_ROW SUM/AVG swaps the operands (see @ref reduce_swaps_operands). The register formats stay with the registers: the
 *       scaler is converted to the SrcA format and the data to the SrcB format.
 * @note PoolType::MIN is rejected here. Nothing reduces without this init, so that closes the whole
 *       FPU path to it.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION, bool EN_32BIT_DEST, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_reduce_init_(const DataFormat src_a_format, const DataFormat src_b_format, const TensorShape tensor_shape)
{
    constexpr MathFidelity FIDELITY = POOL_TYPE == PoolType::MAX ? MathFidelity::LoFi : MATH_FIDELITY_TYPE;
    // There is no min-pool instruction - the FPU has GMPOOL (max) and GAPOOL (average/sum) - so MIN
    // would not fail here, it would quietly average, and the unpacker would pad with zero where a
    // min reduce needs +inf. PoolType::MIN exists for the SFPU reduce, which implements it.
    static_assert(
        POOL_TYPE != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    [[maybe_unused]] const auto fidelity = math_fidelity_schedule<FIDELITY>(src_a_format, src_b_format);

    if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW && POOL_TYPE == PoolType::MAX)
    {
        _llk_math_reduce_row_max_addrmod_();
        _llk_math_reduce_row_max_mop_config_(tensor_shape);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW)
    {
        _llk_math_reduce_row_sum_addrmod_(fidelity, tensor_shape);
        _llk_math_reduce_row_sum_mop_config_(fidelity, tensor_shape);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_COL)
    {
        _llk_math_reduce_col_addrmod_(fidelity);
        _llk_math_reduce_col_mop_config_<POOL_TYPE>(fidelity, tensor_shape);
    }
    else if constexpr (REDUCE_DIMENSION == ReduceDim::REDUCE_SCALAR)
    {
        const auto final_fidelity = math_fidelity_schedule<FIDELITY>(dest_src_format<EN_32BIT_DEST>(src_a_format), src_b_format);
        _llk_math_reduce_scalar_addrmod_(fidelity, final_fidelity);
        _llk_math_reduce_scalar_mop_config_<POOL_TYPE, EN_32BIT_DEST>(fidelity, final_fidelity, tensor_shape);
    }

    // For face_r_dim >= 8, dest is dense with tiles. For face_r_dim < 8, dest is sparse with tiles and tiles are placed every 8 rows.
    // If num_rows_per_tile is less than that of face_r_dim = 8, replace it to ensure face_r_dim = 8 sparse layout.
    _set_tile_shape_idx_gpr_(find_max(FACE_R_DIM, tensor_shape.face_r_dim * tensor_shape.total_num_faces()));

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Perform a reduce operation on one tile: consumes the SrcA and SrcB dvalid of the tile.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Sets the reduce dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tile_idx: Tile index into the destination register. If dest reg in 16-bit mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in
 * full mode. If dest reg in 32-bit mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 * @param tensor_shape: Tile shape; must match init.
 * @note Call @ref _llk_math_reduce_init_ with matching template args before this function.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION>
inline void _llk_math_reduce_(const std::uint32_t tile_idx, [[maybe_unused]] const TensorShape& tensor_shape)
{
    _set_dst_write_addr_by_rows_(tile_idx);

    // The MOP ends by clearing the SrcA/SrcB dvalid and resetting the counters
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
