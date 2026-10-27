// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
using namespace ckernel;

/**
 * @brief Unpack one hardware tile of reduce data into the data source register.
 *
 * @tparam SWAP_OPERANDS: Data goes to SrcB (UNPACKER1) instead of SrcA (UNPACKER0), see @ref reduce_swaps_operands.
 * @param buf_desc_id: Buffer descriptor ID of the data.
 * @param dst_tile_idx_inc: Register destination tile index increment applied after the unpack.
 * @param set_dvalid: Set the data valid of the register after the unpack.
 */
template <bool SWAP_OPERANDS>
inline void _reduce_unpack_data_(const std::uint32_t buf_desc_id, const std::uint32_t dst_tile_idx_inc, const std::uint32_t set_dvalid)
{
    if constexpr (SWAP_OPERANDS)
    {
        TT_UNPACR1_TILE_INC(dst_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id, set_dvalid);
    }
    else
    {
        TT_UNPACR0_TILE_INC(dst_tile_idx_inc, 1 /*Src Tile Idx*/, buf_desc_id, set_dvalid);
    }
}

/**
 * @brief Builds the MOP for unpack reduce operations.
 *
 * Unpacks one face of the scaler, then the whole data tile with a single dvalid. buf_desc_id_0 holds the data,
 * buf_desc_id_1 the scaler. The data goes to SrcA and the scaler to SrcB, except for REDUCE_ROW SUM/AVG
 * (see @ref reduce_swaps_operands), which unpacks the data to SrcB and the scaler to SrcA.
 * Every data face lands in its own MAX_FACE_R_DIM-row slot; for faces shorter than that, the data register is first
 * cleared to the pool identity.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc.
 * @param num_tiles: Number of data tiles to unpack at a time; the scaler face is unpacked once per data tile.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION>
inline void _llk_unpack_reduce_mop_config_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape& tensor_shape, const std::uint32_t num_tiles)
{
    constexpr bool swap_operands          = reduce_swaps_operands<POOL_TYPE, REDUCE_DIMENSION>();
    constexpr std::uint32_t data_unpacker = swap_operands ? p_unpacr::UNP_B : p_unpacr::UNP_A;
    constexpr std::uint32_t clr_mode      = (POOL_TYPE == PoolType::MAX) ? p_unpacr::UNP_CLRSRC_NEGINF : p_unpacr::UNP_CLRSRC_ZERO;

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool is_tiny            = tensor_shape.face_r_dim < MAX_FACE_R_DIM;

    // One face of the scaler. A four-face buffer descriptor holds one hardware tile per tile, other shapes one per face.
    std::uint32_t unpack_scaler;
    if (num_faces == NUM_FACES)
    {
        unpack_scaler = swap_operands ? TT_OP_UNPACR0_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/)
                                      : TT_OP_UNPACR1_FACE_INC(0, 0, 0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    }
    else
    {
        unpack_scaler =
            swap_operands ? TT_OP_UNPACR0_TILE_INC(0, 0, buf_desc_id_1, 1 /*Set Dvalid*/) : TT_OP_UNPACR1_TILE_INC(0, 0, buf_desc_id_1, 1 /*Set Dvalid*/);
    }

    // Batched tiles of one or two faces restart the register destination counter for every tile
    const bool reset_dst_idx           = num_tiles > 1 && num_faces != NUM_FACES;
    const std::uint32_t replay_buf_len = (reset_dst_idx ? 1 : 0) + (is_tiny ? 1 : 0) + ((num_faces == NUM_FACES) ? 1 : num_faces);

    load_replay_buf(
        0,
        replay_buf_len,
        false,
        0,
        0,
        [&tensor_shape, buf_desc_id_0, num_faces, is_tiny, reset_dst_idx]
        {
            if (reset_dst_idx)
            {
                TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, data_unpacker, 0);
            }

            if (is_tiny)
            {
                TTI_UNPACR_NOP(data_unpacker, 0, p_unpacr::UNP_STALL_UNP_WR, 0 /* clear curr bank */, clr_mode, p_unpacr::UNP_CLRSRC_ZERO /* UNP_CLR_SRC */);
            }

            if (num_faces == NUM_FACES)
            {
                // The whole 32x32 tile in one unpack
                _reduce_unpack_data_<swap_operands>(buf_desc_id_0, 0, 1 /*Set Dvalid*/);
            }
            else
            {
                // One hardware tile per face; skip to the next MAX_FACE_R_DIM-row slot after each face
                const std::uint32_t dst_tile_idx_inc = MAX_FACE_R_DIM / tensor_shape.face_r_dim;
                for (std::uint32_t face = 0; face < num_faces - 1; face++)
                {
                    _reduce_unpack_data_<swap_operands>(buf_desc_id_0, dst_tile_idx_inc, 0 /*Set Dvalid*/);
                }
                _reduce_unpack_data_<swap_operands>(buf_desc_id_0, 0, 1 /*Set Dvalid*/);
            }
        });

    ckernel_template temp(num_tiles, 1 /*inner loop*/, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
    temp.set_start_op(unpack_scaler);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes the unpacker for reduce operations.
 *
 * Validates the tensor shape, programs the SrcA transpose for REDUCE_ROW, then programs the MOP for
 * unpacking the whole data tile and a single face of the scaler with one dvalid each. buf_desc_id_0 holds the data,
 * buf_desc_id_1 the scaler. For REDUCE_ROW SUM/AVG the data goes to SrcB and the scaler to SrcA
 * (see @ref reduce_swaps_operands), for all other reduces the data goes to SrcA and the scaler to SrcB.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param buf_desc_id_0/1: The buffer descriptor ID where the buffer information is
 *        stored in the buffer descriptor table, values = 0 - 16
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc.
 * @param num_tiles: Number of data tiles to unpack at a time; the scaler will only have its first face unpacked.
 * @note On the math thread, pair with @ref _llk_math_reduce_init_ (T1); on the pack thread, pair with @ref _llk_pack_reduce_mask_config_ (T2).
 * @note @ref _llk_unpack_reduce_ is the matching execute call on this thread.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION>
inline void _llk_unpack_reduce_init_(
    const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape tensor_shape, const std::uint32_t num_tiles = NUM_TILES)
{
    static_assert(
        POOL_TYPE != PoolType::MIN,
        "The FPU reduce has no MIN: the unpacker's partial-face fill clears SrcA to zero for every "
        "pool but MAX, which a min reduce would lose to. Use the SFPU reduce instead "
        "(ckernel_sfpu_reduce.h::calculate_reduce).");

    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, (REDUCE_DIMENSION == ReduceDim::REDUCE_ROW));
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);
    _llk_unpack_reduce_mop_config_<POOL_TYPE, REDUCE_DIMENSION>(buf_desc_id_0, buf_desc_id_1, tensor_shape, num_tiles);
}

/**
 * @brief Unpacks operands for reduce kernels into SrcA and SrcB.
 *
 * @tparam POOL_TYPE: Type of reduce pool op, values = <MAX/SUM/AVG>
 * @tparam REDUCE_DIMENSION: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param start_l1_tile_idx_0/1: Start tile index into the L1 buffer of the data (0) and the scaler (1).
 * @param tensor_shape: Contains all the information of the tile shape: num faces, face row/col dim, etc.
 * @note Call @ref _llk_unpack_reduce_init_ with matching template args before this function.
 */
template <PoolType POOL_TYPE, ReduceDim REDUCE_DIMENSION>
inline void _llk_unpack_reduce_(const std::uint32_t start_l1_tile_idx_0, const std::uint32_t start_l1_tile_idx_1, const TensorShape& tensor_shape)
{
    // RT: for the best performance, setting counters should be placed in a REPLAY buffer
    // in the mop_config, but for back compatibility with APIs, the counter functions must
    // be programmable with users input offset idx

    // Reset Dest counters for Unpacker0/1 to 0
    // Set Source counter to L1 base + offset
    constexpr bool swap_operands = reduce_swaps_operands<POOL_TYPE, REDUCE_DIMENSION>();

    const std::uint32_t hw_tiles_per_sw_tile = (tensor_shape.total_num_faces() == NUM_FACES) ? 1 : tensor_shape.total_num_faces();
    const std::uint32_t l1_idx_data          = start_l1_tile_idx_0 * hw_tiles_per_sw_tile;
    const std::uint32_t l1_idx_scaler        = start_l1_tile_idx_1 * hw_tiles_per_sw_tile;

    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, swap_operands ? l1_idx_scaler : l1_idx_data);
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, swap_operands ? l1_idx_data : l1_idx_scaler);

    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
