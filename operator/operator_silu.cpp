#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "npu_dma.h"
#include "npu_rvv.h"
#include "operator_silu.h"

// SILU Operator
// Formula: x / (1 + exp(-x))
void user_operator_silu(int* cycle_cnt, int Rows, int Cols, float* data_ddr, float* rslt_ddr) {
    printf("\nRunning User Operator: SILU [%d, %d]...\n", Rows, Cols);
    // 
    int max_cycles = 0;
    int Bc = Cols / NPU_NUM_CLUSTERS; // Tensor Parallel: 4 clusters split columns
    int Br = 32; // 32 rows per tile

    rvv_configure(1, NPU_RVV_VLEN_DLEN); 
        
    // Process each cluster
    for (int c = 0; c < NPU_NUM_CLUSTERS; c++) {
        int cluster_cycles = 0;
        sram_reset();
        
        // Process rows in tiles
        for (int r = 0; r < Rows; r += Br) {
            int Br_r = (r + Br <= Rows) ? Br : (Rows - r);
            
            // Allocate SRAM for this tile
            size_t tile_size = Br_r * Bc * sizeof(float);
            
            // Input tile buffer in SRAM
            float* sram_x = (float*)sram_malloc(tile_size);
            
            if (!sram_x) {
                printf("OOM in SRAM for cluster %d tile\n", c);
                free(data_ddr);
                return;
            }
            
            // 1. DMA Load: DDR -> SRAM (TODO: modify loop in stride=DMA_BUS_width）
            int64_t Bc_64 = Bc;
            for (int rr = 0; rr < Br_r; rr++) {
                int row_idx = r + rr;
                float* ddr_slice = data_ddr + row_idx * Cols + c * Bc;
                float* sram_slice = sram_x + rr * Bc;
                npu_tensor_t t_ddr_slice = create_tensor(ddr_slice, 1, &Bc_64, NPU_TYPE_FP32);
                npu_tensor_t t_sram_slice = create_tensor(sram_slice, 1, &Bc_64, NPU_TYPE_FP32);
                dma_copy(&t_sram_slice, &t_ddr_slice, &cluster_cycles);
            }
            //printf("DMA load: %d cycles\n", cluster_cycles);

            // 2. Compute SILU: x / (1 + exp(-x)) (TODO: modify loop in stride=SIMD_width)
            int64_t tile_dims[] = {Bc, Br_r};
            npu_tensor_t t_sram_x = create_tensor(sram_x, 2, tile_dims, NPU_TYPE_FP32);
            
            // Load from SRAM to RVV Register
            float* x_ptr = (float*)malloc(tile_size); // "RVV Register" (TODO:npu_rvv_reg.cpp)
            if (!x_ptr) { printf("OOM x\n"); free(data_ddr); return; }
            npu_tensor_t t_x = create_tensor(x_ptr, 2, tile_dims, NPU_TYPE_FP32);
            
            rvv_vle32_v(&t_x, &t_sram_x, &cluster_cycles);
            
            // neg_x = -x
            float* neg_x_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_neg_x = create_tensor(neg_x_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfneg(&t_neg_x, &t_x, &cluster_cycles);

            // --- BEGIN ggml_v_expf_m2 INLINE ---
            // Input: t_neg_x
            // Result will be in t_exp_res

            // const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
            float* r_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_r = create_tensor(r_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmv_v_f(&t_r, 12582912.0f, &cluster_cycles); // 0x1.8p23f

            // const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
            // Note: x here is t_neg_x
            // z = r + (magic * neg_x)
            float* z_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_z = create_tensor(z_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vle(&t_z, &t_r, &cluster_cycles); // Init z with r
            rvv_vfmacc_vf(&t_z, 1.4426950408889634f, &t_neg_x, &cluster_cycles); // 0x1.715476p+0f

            // const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
            float* n_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_n = create_tensor(n_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vsub(&t_n, &t_z, &t_r, &cluster_cycles);

            // const vfloat32m2_t b = __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
            // Inner: tmp = x - (0x1.62e4p-1f * n)  => tmp = neg_x - (C1 * n)
            float* b_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_b = create_tensor(b_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vle(&t_b, &t_neg_x, &cluster_cycles); // Init b with neg_x
            rvv_vfnmsac_vf(&t_b, 0.6931471805599453f, &t_n, &cluster_cycles); // 0x1.62e4p-1f

            // Outer: b = tmp - (0x1.7f7d1cp-20f * n)
            rvv_vfnmsac_vf(&t_b, 1.428606820309417e-06f, &t_n, &cluster_cycles); // 0x1.7f7d1cp-20f

            // const vuint32m2_t e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
            float* e_ptr = (float*)malloc(tile_size); // Stored as float/uint mix in model
            npu_tensor_t t_e = create_tensor(e_ptr, 2, tile_dims, NPU_TYPE_INT32); // Treat as INT32
            
            float* z_as_int_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_z_as_int = create_tensor(z_as_int_ptr, 2, tile_dims, NPU_TYPE_INT32);
            rvv_vreinterpret(&t_z_as_int, &t_z, &cluster_cycles);
            rvv_vsll_vx(&t_e, &t_z_as_int, 23, &cluster_cycles);

            // const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
            float* k_int_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_k_int = create_tensor(k_int_ptr, 2, tile_dims, NPU_TYPE_INT32);
            rvv_vadd_vx(&t_k_int, &t_e, 0x3f800000, &cluster_cycles);

            float* k_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_k = create_tensor(k_ptr, 2, tile_dims, NPU_TYPE_FP32); // Final float
            rvv_vreinterpret(&t_k, &t_k_int, &cluster_cycles);

            // const vbool16_t c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
            float* abs_n_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_abs_n = create_tensor(abs_n_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfabs(&t_abs_n, &t_n, &cluster_cycles);

            float* c_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_c = create_tensor(c_ptr, 2, tile_dims, NPU_TYPE_FP32); // Mask stored as float (1.0/0.0)
            rvv_vmfgt(&t_c, &t_abs_n, 126.0f, &cluster_cycles);

            // const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
            float* u_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_u = create_tensor(u_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmul(&t_u, &t_b, &t_b, &cluster_cycles);

            // Polynomial j
            // t1 = 0x1.fffdb6p-2f (0.49998)
            float* j_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_j = create_tensor(j_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmv_v_f(&t_j, 0.49998000264167786f, &cluster_cycles); 

            // t1 = vfmacc(t1, 0x1.555e66p-3f, b) -> 0.16666...
            rvv_vfmacc_vf(&t_j, 0.16666552424430847f, &t_b, &cluster_cycles);

            // t2 = 0x1.573e2ep-5f (0.04191)
            float* p2_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_p2 = create_tensor(p2_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmv_v_f(&t_p2, 0.041917506605386734f, &cluster_cycles);

            // t2 = vfmacc(t2, 0x1.0e4020p-7f, b) -> 0.00824...
            rvv_vfmacc_vf(&t_p2, 0.0082455575466156006f, &t_b, &cluster_cycles);

            // j = vfmacc(j, t2, u) -> j + t2 * u
            rvv_vfmacc(&t_j, &t_p2, &t_u, &cluster_cycles);

            // t3 = b * 0x1.ffffecp-1f (0.999999...)
            float* p3_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_p3 = create_tensor(p3_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmul_vf(&t_p3, &t_b, 0.9999998807907104f, &cluster_cycles);

            // j = vfmacc(t3, j, u) -> t3 + j * u
            rvv_vfmacc(&t_p3, &t_j, &t_u, &cluster_cycles);
            
            // Final j is in t_p3.
            npu_tensor_t* t_final_j = &t_p3;
            // Final Result Variable for EXP
            float* exp_res_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_exp_res = create_tensor(exp_res_ptr, 2, tile_dims, NPU_TYPE_FP32);

            // if (!vcpop(c))
            int count_c = rvv_vcpop_m(&t_c, &cluster_cycles);
            
            if (count_c == 0) {
                // return k + j * k
                rvv_vfmacc(&t_k, t_final_j, &t_k, &cluster_cycles);
                rvv_vle(&t_exp_res, &t_k, &cluster_cycles);
            } else {
                // dm = n <= 0.0
                float* dm_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_dm = create_tensor(dm_ptr, 2, tile_dims, NPU_TYPE_FP32);
                rvv_vmfle(&t_dm, &t_n, 0.0f, &cluster_cycles);

                // d = vmerge_vxm(vmv_v_x(0), 0x82000000, dm)
                float* d_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_d = create_tensor(d_ptr, 2, tile_dims, NPU_TYPE_INT32);
                
                float* d_base_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_d_base = create_tensor(d_base_ptr, 2, tile_dims, NPU_TYPE_INT32);
                rvv_vmv_v_x(&t_d_base, 0, &cluster_cycles);

                rvv_vmerge_vxm(&t_d, &t_d_base, 0x82000000, &t_dm, &cluster_cycles);

                // s1 = d + 0x7f000000
                float* s1_int_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_s1_int = create_tensor(s1_int_ptr, 2, tile_dims, NPU_TYPE_INT32);
                rvv_vadd_vx(&t_s1_int, &t_d, 0x7f000000, &cluster_cycles);
                
                float* s1_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_s1 = create_tensor(s1_ptr, 2, tile_dims, NPU_TYPE_FP32);
                rvv_vreinterpret(&t_s1, &t_s1_int, &cluster_cycles);

                // s2 = e - d
                float* s2_int_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_s2_int = create_tensor(s2_int_ptr, 2, tile_dims, NPU_TYPE_INT32);
                rvv_vsub_vv(&t_s2_int, &t_e, &t_d, &cluster_cycles);
                
                float* s2_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_s2 = create_tensor(s2_ptr, 2, tile_dims, NPU_TYPE_FP32);
                rvv_vreinterpret(&t_s2, &t_s2_int, &cluster_cycles);

                // r1 calculation
                // part1 = k + k*j
                float* r1_p1_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_r1_p1 = create_tensor(r1_p1_ptr, 2, tile_dims, NPU_TYPE_FP32);
                rvv_vle(&t_r1_p1, &t_k, &cluster_cycles);
                rvv_vfmacc(&t_r1_p1, t_final_j, &t_k, &cluster_cycles);

                // part2 = (s2 + s2*j) * s1
                float* r1_p2_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_r1_p2 = create_tensor(r1_p2_ptr, 2, tile_dims, NPU_TYPE_FP32);
                rvv_vle(&t_r1_p2, &t_s2, &cluster_cycles);
                rvv_vfmacc(&t_r1_p2, &t_s2, t_final_j, &cluster_cycles); // s2+s2*j
                rvv_vfmul(&t_r1_p2, &t_r1_p2, &t_s1, &cluster_cycles);

                // r1 = merge(part1, part2, c) (true=part2, false=part1)
                float* r1_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_r1 = create_tensor(r1_ptr, 2, tile_dims, NPU_TYPE_FP32);
                rvv_vmerge(&t_r1, &t_r1_p2, &t_r1_p1, &t_c, &cluster_cycles);

                // Final merge
                // term_sq = s1 * s1
                float* term_sq_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_term_sq = create_tensor(term_sq_ptr, 2, tile_dims, NPU_TYPE_FP32);
                rvv_vfmul(&t_term_sq, &t_s1, &t_s1, &cluster_cycles);

                // mask = abs(n) > 192.0
                float* mask_big_ptr = (float*)malloc(tile_size);
                npu_tensor_t t_mask_big = create_tensor(mask_big_ptr, 2, tile_dims, NPU_TYPE_FP32);
                rvv_vmfgt(&t_mask_big, &t_abs_n, 192.0f, &cluster_cycles);

                // result = merge(r1, term_sq, mask_big)
                rvv_vmerge(&t_exp_res, &t_term_sq, &t_r1, &t_mask_big, &cluster_cycles);

                free(dm_ptr);
                free(d_ptr);
                free(d_base_ptr);
                free(s1_ptr);
                free(s1_int_ptr);
                free(s2_ptr);
                free(s2_int_ptr);
                free(r1_p1_ptr);
                free(r1_p2_ptr);
                free(r1_ptr);
                free(term_sq_ptr);
                free(mask_big_ptr);
            }

            // --- END EXP ---

            // 1.0 + exp(-x)
            float* one_plus_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_one_plus = create_tensor(one_plus_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfadd_vf(&t_one_plus, &t_exp_res, 1.0f, &cluster_cycles);

            // x / (1 + exp(-x))
            // result is in Register (t_result was allocated in 'reg' space)
            float* reg_result_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_result = create_tensor(reg_result_ptr, 2, tile_dims, NPU_TYPE_FP32);
            
            rvv_vdiv(&t_result, &t_x, &t_one_plus, &cluster_cycles);

            // Store Register -> SRAM
            float* sram_result_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_sram_result = create_tensor(sram_result_ptr, 2, tile_dims, NPU_TYPE_FP32);
            
            rvv_vse32_v(&t_sram_result, &t_result, &cluster_cycles);

            //printf("exp calc done: %d cycles\n", cluster_cycles);

            // 3. DMA Store: SRAM -> DDR (TODO: modify loop in stride=DMA_BUS_width)
            for (int rr = 0; rr < Br_r; rr++) {
                int row_idx = r + rr;
                float* ddr_slice = rslt_ddr + row_idx * Cols + c * Bc;
                float* sram_slice = sram_result_ptr + rr * Bc;
                npu_tensor_t t_ddr_slice = create_tensor(ddr_slice, 1, &Bc_64, NPU_TYPE_FP32);
                npu_tensor_t t_sram_slice = create_tensor(sram_slice, 1, &Bc_64, NPU_TYPE_FP32);
                dma_copy(&t_ddr_slice, &t_sram_slice, &cluster_cycles);
            }

            //printf("DMA store: %d cycles\n", cluster_cycles);

            free(x_ptr);
            free(neg_x_ptr);
            free(r_ptr);
            free(z_ptr);
            free(z_as_int_ptr);
            free(n_ptr);
            free(b_ptr);
            free(e_ptr);
            free(k_ptr);
            free(k_int_ptr);
            free(abs_n_ptr);
            free(c_ptr);
            free(u_ptr);
            free(j_ptr);
            free(p2_ptr);
            free(p3_ptr);
            free(exp_res_ptr);
            free(one_plus_ptr);
            free(reg_result_ptr);
            free(sram_result_ptr);
            
            sram_reset(); // Reset for next batch
        }
        //printf("Cluster cycles: %d\n", cluster_cycles);
        if (cluster_cycles > max_cycles) max_cycles = cluster_cycles;
    }
    
    *cycle_cnt += max_cycles;
    
}

