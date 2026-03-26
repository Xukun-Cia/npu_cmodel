#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "npu_dma.h"
#include "npu_rvv.h"
#include "operator_gelu.h"

// GELU constants
#define GELU_COEF_A    0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

// Helper function to compute exp(x) using SIMD
// This is similar to the exp implementation in SILU
static void rvv_exp(npu_tensor_t* dst, const npu_tensor_t* src, int* t, size_t tile_size, int64_t* tile_dims) {
    // Input: src
    // Result will be in dst
    
    // const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    float* r_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_r = create_tensor(r_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfmv_v_f(&t_r, 12582912.0f, t); // 0x1.8p23f

    // const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    // z = r + (magic * x)
    float* z_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_z = create_tensor(z_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vle(&t_z, &t_r, t); // Init z with r
    rvv_vfmacc_vf(&t_z, 1.4426950408889634f, src, t); // 0x1.715476p+0f

    // const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    float* n_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_n = create_tensor(n_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vsub(&t_n, &t_z, &t_r, t);

    // const vfloat32m2_t b = __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    // Inner: tmp = x - (0x1.62e4p-1f * n)
    float* b_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_b = create_tensor(b_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vle(&t_b, src, t); // Init b with src
    rvv_vfnmsac_vf(&t_b, 0.6931471805599453f, &t_n, t); // 0x1.62e4p-1f

    // Outer: b = tmp - (0x1.7f7d1cp-20f * n)
    rvv_vfnmsac_vf(&t_b, 1.428606820309417e-06f, &t_n, t); // 0x1.7f7d1cp-20f

    // const vuint32m2_t e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    float* e_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_e = create_tensor(e_ptr, 2, tile_dims, NPU_TYPE_INT32);
    
    float* z_as_int_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_z_as_int = create_tensor(z_as_int_ptr, 2, tile_dims, NPU_TYPE_INT32);
    rvv_vreinterpret(&t_z_as_int, &t_z, t);
    rvv_vsll_vx(&t_e, &t_z_as_int, 23, t);

    // const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    float* k_int_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_k_int = create_tensor(k_int_ptr, 2, tile_dims, NPU_TYPE_INT32);
    rvv_vadd_vx(&t_k_int, &t_e, 0x3f800000, t);

    float* k_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_k = create_tensor(k_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vreinterpret(&t_k, &t_k_int, t);

    // const vbool16_t c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    float* abs_n_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_abs_n = create_tensor(abs_n_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfabs(&t_abs_n, &t_n, t);

    float* c_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_c = create_tensor(c_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vmfgt(&t_c, &t_abs_n, 126.0f, t);

    // const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    float* u_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_u = create_tensor(u_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfmul(&t_u, &t_b, &t_b, t);

    // Polynomial j
    float* j_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_j = create_tensor(j_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfmv_v_f(&t_j, 0.49998000264167786f, t); // 0x1.fffdb6p-2f

    rvv_vfmacc_vf(&t_j, 0.16666552424430847f, &t_b, t); // 0x1.555e66p-3f

    float* p2_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_p2 = create_tensor(p2_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfmv_v_f(&t_p2, 0.041917506605386734f, t); // 0x1.573e2ep-5f

    rvv_vfmacc_vf(&t_p2, 0.0082455575466156006f, &t_b, t); // 0x1.0e4020p-7f

    rvv_vfmacc(&t_j, &t_p2, &t_u, t);

    float* p3_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_p3 = create_tensor(p3_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfmul_vf(&t_p3, &t_b, 0.9999998807907104f, t); // 0x1.ffffecp-1f

    rvv_vfmacc(&t_p3, &t_j, &t_u, t);
    
    npu_tensor_t* t_final_j = &t_p3;

    // if (!vcpop(c))
    int count_c = rvv_vcpop_m(&t_c, t);
    
    if (count_c == 0) {
        // return k + j * k
        rvv_vfmacc(&t_k, t_final_j, &t_k, t);
        rvv_vle(dst, &t_k, t);
    } else {
        // dm = n <= 0.0
        float* dm_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_dm = create_tensor(dm_ptr, 2, tile_dims, NPU_TYPE_FP32);
        rvv_vmfle(&t_dm, &t_n, 0.0f, t);

        // d = vmerge_vxm(vmv_v_x(0), 0x82000000, dm)
        float* d_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_d = create_tensor(d_ptr, 2, tile_dims, NPU_TYPE_INT32);
        
        float* d_base_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_d_base = create_tensor(d_base_ptr, 2, tile_dims, NPU_TYPE_INT32);
        rvv_vmv_v_x(&t_d_base, 0, t);

        rvv_vmerge_vxm(&t_d, &t_d_base, 0x82000000, &t_dm, t);

        // s1 = d + 0x7f000000
        float* s1_int_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_s1_int = create_tensor(s1_int_ptr, 2, tile_dims, NPU_TYPE_INT32);
        rvv_vadd_vx(&t_s1_int, &t_d, 0x7f000000, t);
        
        float* s1_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_s1 = create_tensor(s1_ptr, 2, tile_dims, NPU_TYPE_FP32);
        rvv_vreinterpret(&t_s1, &t_s1_int, t);

        // s2 = e - d
        float* s2_int_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_s2_int = create_tensor(s2_int_ptr, 2, tile_dims, NPU_TYPE_INT32);
        rvv_vsub_vv(&t_s2_int, &t_e, &t_d, t);
        
        float* s2_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_s2 = create_tensor(s2_ptr, 2, tile_dims, NPU_TYPE_FP32);
        rvv_vreinterpret(&t_s2, &t_s2_int, t);

        // r1 calculation
        float* r1_p1_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_r1_p1 = create_tensor(r1_p1_ptr, 2, tile_dims, NPU_TYPE_FP32);
        rvv_vle(&t_r1_p1, &t_k, t);
        rvv_vfmacc(&t_r1_p1, t_final_j, &t_k, t);

        float* r1_p2_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_r1_p2 = create_tensor(r1_p2_ptr, 2, tile_dims, NPU_TYPE_FP32);
        rvv_vle(&t_r1_p2, &t_s2, t);
        rvv_vfmacc(&t_r1_p2, &t_s2, t_final_j, t);
        rvv_vfmul(&t_r1_p2, &t_r1_p2, &t_s1, t);

        float* r1_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_r1 = create_tensor(r1_ptr, 2, tile_dims, NPU_TYPE_FP32);
        rvv_vmerge(&t_r1, &t_r1_p2, &t_r1_p1, &t_c, t);

        float* term_sq_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_term_sq = create_tensor(term_sq_ptr, 2, tile_dims, NPU_TYPE_FP32);
        rvv_vfmul(&t_term_sq, &t_s1, &t_s1, t);

        float* mask_big_ptr = (float*)malloc(tile_size);
        npu_tensor_t t_mask_big = create_tensor(mask_big_ptr, 2, tile_dims, NPU_TYPE_FP32);
        rvv_vmfgt(&t_mask_big, &t_abs_n, 192.0f, t);

        rvv_vmerge(dst, &t_term_sq, &t_r1, &t_mask_big, t);

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

    // Free temporary buffers
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
}

// Helper function to compute tanh(x) using SIMD
// tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
static void rvv_tanh(npu_tensor_t* dst, const npu_tensor_t* src, int* t, size_t tile_size, int64_t* tile_dims) {
    // Compute 2*x
    float* two_x_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_two_x = create_tensor(two_x_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfmul_vf(&t_two_x, src, 2.0f, t);
    
    // Compute exp(2*x)
    float* exp_2x_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_exp_2x = create_tensor(exp_2x_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_exp(&t_exp_2x, &t_two_x, t, tile_size, tile_dims);
    
    // Compute exp(2*x) - 1
    float* exp_minus_one_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_exp_minus_one = create_tensor(exp_minus_one_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfadd_vf(&t_exp_minus_one, &t_exp_2x, -1.0f, t);
    
    // Compute exp(2*x) + 1
    float* exp_plus_one_ptr = (float*)malloc(tile_size);
    npu_tensor_t t_exp_plus_one = create_tensor(exp_plus_one_ptr, 2, tile_dims, NPU_TYPE_FP32);
    rvv_vfadd_vf(&t_exp_plus_one, &t_exp_2x, 1.0f, t);
    
    // Compute tanh = (exp(2*x) - 1) / (exp(2*x) + 1)
    rvv_vdiv(dst, &t_exp_minus_one, &t_exp_plus_one, t);
    
    // Free temporary buffers
    free(two_x_ptr);
    free(exp_2x_ptr);
    free(exp_minus_one_ptr);
    free(exp_plus_one_ptr);
}

// GELU Operator
// Formula: 0.5 * x * (1 + tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x)))
void user_operator_gelu(int* cycle_cnt, int Rows, int Cols, float* data_ddr, float* rslt_ddr) {
    printf("\nRunning User Operator: GELU [%d, %d]...\n", Rows, Cols);
    
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
            
            // 1. DMA Load: DDR -> SRAM
            int64_t Bc_64 = Bc;
            for (int rr = 0; rr < Br_r; rr++) {
                int row_idx = r + rr;
                float* ddr_slice = data_ddr + row_idx * Cols + c * Bc;
                float* sram_slice = sram_x + rr * Bc;
                npu_tensor_t t_ddr_slice = create_tensor(ddr_slice, 1, &Bc_64, NPU_TYPE_FP32);
                npu_tensor_t t_sram_slice = create_tensor(sram_slice, 1, &Bc_64, NPU_TYPE_FP32);
                dma_copy(&t_sram_slice, &t_ddr_slice, &cluster_cycles);
            }

            // 2. Compute GELU: 0.5 * x * (1 + tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x)))
            int64_t tile_dims[] = {Bc, Br_r};
            npu_tensor_t t_sram_x = create_tensor(sram_x, 2, tile_dims, NPU_TYPE_FP32);
            
            // Load from SRAM to RVV Register
            float* x_ptr = (float*)malloc(tile_size);
            if (!x_ptr) { printf("OOM x\n"); free(data_ddr); return; }
            npu_tensor_t t_x = create_tensor(x_ptr, 2, tile_dims, NPU_TYPE_FP32);
            
            rvv_vle32_v(&t_x, &t_sram_x, &cluster_cycles);
            
            // Step 1: x_sq = x * x
            float* x_sq_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_x_sq = create_tensor(x_sq_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmul(&t_x_sq, &t_x, &t_x, &cluster_cycles);
            
            // Step 2: coef_x_sq = GELU_COEF_A * x_sq
            float* coef_x_sq_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_coef_x_sq = create_tensor(coef_x_sq_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmul_vf(&t_coef_x_sq, &t_x_sq, GELU_COEF_A, &cluster_cycles);
            
            // Step 3: one_plus_coef = 1.0f + coef_x_sq
            float* one_plus_coef_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_one_plus_coef = create_tensor(one_plus_coef_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfadd_vf(&t_one_plus_coef, &t_coef_x_sq, 1.0f, &cluster_cycles);
            
            // Step 4: x_times_one_plus = x * one_plus_coef
            float* x_times_one_plus_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_x_times_one_plus = create_tensor(x_times_one_plus_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmul(&t_x_times_one_plus, &t_x, &t_one_plus_coef, &cluster_cycles);
            
            // Step 5: sqrt_2_over_pi_x = SQRT_2_OVER_PI * x_times_one_plus
            float* sqrt_2_over_pi_x_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_sqrt_2_over_pi_x = create_tensor(sqrt_2_over_pi_x_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmul_vf(&t_sqrt_2_over_pi_x, &t_x_times_one_plus, SQRT_2_OVER_PI, &cluster_cycles);
            
            // Step 6: tanh_result = tanh(sqrt_2_over_pi_x)
            float* tanh_result_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_tanh_result = create_tensor(tanh_result_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_tanh(&t_tanh_result, &t_sqrt_2_over_pi_x, &cluster_cycles, tile_size, tile_dims);
            
            // Step 7: one_plus_tanh = 1.0f + tanh_result
            float* one_plus_tanh_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_one_plus_tanh = create_tensor(one_plus_tanh_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfadd_vf(&t_one_plus_tanh, &t_tanh_result, 1.0f, &cluster_cycles);
            
            // Step 8: result = 0.5f * x * one_plus_tanh
            float* reg_result_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_result = create_tensor(reg_result_ptr, 2, tile_dims, NPU_TYPE_FP32);
            rvv_vfmul(&t_result, &t_x, &t_one_plus_tanh, &cluster_cycles);
            rvv_vfmul_vf(&t_result, &t_result, 0.5f, &cluster_cycles);

            // Store Register -> SRAM
            float* sram_result_ptr = (float*)malloc(tile_size);
            npu_tensor_t t_sram_result = create_tensor(sram_result_ptr, 2, tile_dims, NPU_TYPE_FP32);
            
            rvv_vse32_v(&t_sram_result, &t_result, &cluster_cycles);

            // 3. DMA Store: SRAM -> DDR
            for (int rr = 0; rr < Br_r; rr++) {
                int row_idx = r + rr;
                float* ddr_slice = rslt_ddr + row_idx * Cols + c * Bc;
                float* sram_slice = sram_result_ptr + rr * Bc;
                npu_tensor_t t_ddr_slice = create_tensor(ddr_slice, 1, &Bc_64, NPU_TYPE_FP32);
                npu_tensor_t t_sram_slice = create_tensor(sram_slice, 1, &Bc_64, NPU_TYPE_FP32);
                dma_copy(&t_ddr_slice, &t_sram_slice, &cluster_cycles);
            }

            // Free temporary buffers
            free(x_ptr);
            free(x_sq_ptr);
            free(coef_x_sq_ptr);
            free(one_plus_coef_ptr);
            free(x_times_one_plus_ptr);
            free(sqrt_2_over_pi_x_ptr);
            free(tanh_result_ptr);
            free(one_plus_tanh_ptr);
            free(reg_result_ptr);
            free(sram_result_ptr);
            
            sram_reset(); // Reset for next batch
        }
        if (cluster_cycles > max_cycles) max_cycles = cluster_cycles;
    }
    
    *cycle_cnt += max_cycles;
}

