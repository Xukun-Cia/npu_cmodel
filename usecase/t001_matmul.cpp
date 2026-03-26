#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "operator/operator_matmul.h"
#include "t001_matmul.h"

// Test function for MatMul operator
// All tests compare against BF16 reference (BF16 multiply + FP32 accumulator)
// This measures precision loss when using FP16/FP8 instead of native BF16
int test_matmul(int M, int N, int K, const char* type) {
    // Validate type parameter
    npu_datatype_t npu_type;
    if (strcmp(type, "fp16") == 0) {
        npu_type = NPU_TYPE_FP16;
        printf(" Case: MATMUL TEST (FP16 vs BF16 Reference)\n");
    } else if (strcmp(type, "bf16") == 0) {
        npu_type = NPU_TYPE_BF16;
        printf(" Case: MATMUL TEST (BF16 vs BF16 Reference)\n");
    } else if (strcmp(type, "fp8") == 0) {
        npu_type = NPU_TYPE_FP8;
        printf(" Case: MATMUL TEST (FP8 vs BF16 Reference)\n");
    } else {
        fprintf(stderr, "ERROR: Invalid type '%s'. Must be 'fp16', 'bf16', or 'fp8'\n", type);
        return 1;
    }
    
    printf("==============================================================\n");
    printf(" Reference: BF16 multiply + FP32 accumulator\n");

    stats_reset();
    sram_init();

    int matmul_M = M;
    int matmul_N = N;
    int matmul_K = K;

    // Generate random BF16 data (used as reference for all types)
    unsigned int seed = time(NULL);
    
    // Generate BF16 random values in FP16-safe range
    // FP16 max is ~65504, so we need to keep values small enough that 
    // matmul results (K accumulations) stay within FP16 range
    // Use exp range 115-125 (actual exp: -12 to -2) for small values
    auto generate_bf16_random = [&seed]() -> float {
        uint8_t sign = rand_r(&seed) & 0x1;
        uint8_t exp = 115 + (rand_r(&seed) % 11);  // exp: -12 to -2, values ~0.0001 to 0.25
        uint8_t mant = rand_r(&seed) & 0x7F;
        uint16_t bf16_val = (sign << 15) | (exp << 7) | mant;
        return bf16_to_fp32(bf16_val);
    };

    // Allocate BF16 reference data
    uint16_t* a_bf16_ref = (uint16_t*)malloc(matmul_M * matmul_K * sizeof(uint16_t));
    uint16_t* b_bf16_ref = (uint16_t*)malloc(matmul_K * matmul_N * sizeof(uint16_t));
    if (!a_bf16_ref || !b_bf16_ref) {
        printf("Failed to allocate BF16 reference memory\n");
        return 1;
    }

    for(int i=0; i<matmul_M*matmul_K; i++) {
        a_bf16_ref[i] = fp32_to_bf16(generate_bf16_random());
    }
    for(int i=0; i<matmul_K*matmul_N; i++) {
        b_bf16_ref[i] = fp32_to_bf16(generate_bf16_random());
    }

    // Calculate BF16 reference result: BF16 multiply + FP32 accumulator
    float* c_fp32_ref = (float*)malloc(matmul_M * matmul_N * sizeof(float));
    if (!c_fp32_ref) {
        printf("Failed to allocate reference result memory\n");
        return 1;
    }
    memset(c_fp32_ref, 0, matmul_M * matmul_N * sizeof(float));
    
    int tile_M = matmul_M / NPU_TPU_ARRAY_HEIGHT;
    int tile_N = matmul_N / NPU_TPU_ARRAY_WIDTH;
    int tile_K = matmul_K / NPU_TPU_ARRAY_WIDTH;
    
    for (int m_idx = 0; m_idx < tile_M; m_idx++) {
        for (int n_idx = 0; n_idx < tile_N; n_idx++) {
            for (int k_idx = 0; k_idx < tile_K; k_idx++) {
                for (int m = 0; m < NPU_TPU_ARRAY_HEIGHT; m++) {
                    for (int n = 0; n < NPU_TPU_ARRAY_WIDTH; n++) {
                        int r = m_idx * NPU_TPU_ARRAY_HEIGHT + m;
                        int c = n_idx * NPU_TPU_ARRAY_WIDTH + n;
                        float sum = c_fp32_ref[r * matmul_N + c];
                        
                        for (int k = 0; k < NPU_TPU_ARRAY_WIDTH; k++) {
                            int k_global = k_idx * NPU_TPU_ARRAY_WIDTH + k;
                            // BF16 multiply + FP32 accumulate
                            float a = bf16_to_fp32(a_bf16_ref[r * matmul_K + k_global]);
                            float b = bf16_to_fp32(b_bf16_ref[k_global * matmul_N + c]);
                            sum += a * b;
                        }
                        
                        c_fp32_ref[r * matmul_N + c] = sum;
                    }
                }
            }
        }
    }
    
    // Convert reference result to BF16
    uint16_t* c_bf16_ref = (uint16_t*)malloc(matmul_M * matmul_N * sizeof(uint16_t));
    for (int i = 0; i < matmul_M * matmul_N; i++) {
        c_bf16_ref[i] = fp32_to_bf16(c_fp32_ref[i]);
    }
    
    if (npu_type == NPU_TYPE_FP16) {
        // Convert BF16 reference data to FP16 for operator input
        uint16_t* a_fp16 = (uint16_t*)malloc(matmul_M * matmul_K * sizeof(uint16_t));
        uint16_t* b_fp16 = (uint16_t*)malloc(matmul_K * matmul_N * sizeof(uint16_t));
        uint16_t* c_fp16 = (uint16_t*)malloc(matmul_M * matmul_N * sizeof(uint16_t));
        if (!a_fp16 || !b_fp16 || !c_fp16) {
            printf("Failed to allocate FP16 memory\n");
            return 1;
        }

        for(int i=0; i<matmul_M*matmul_K; i++) {
            a_fp16[i] = fp32_to_fp16(bf16_to_fp32(a_bf16_ref[i]));
        }
        for(int i=0; i<matmul_K*matmul_N; i++) {
            b_fp16[i] = fp32_to_fp16(bf16_to_fp32(b_bf16_ref[i]));
        }
        
        int fp16_cycles = 0;
        user_operator_matmul(&fp16_cycles, NPU_TYPE_FP16, matmul_M, matmul_N, matmul_K, a_fp16, b_fp16, c_fp16);

        printf("\nChecking first 10 results (FP16 result vs BF16 reference):\n");
        
        float max_rel_err = 0.0f;
        float min_rel_err = 1e9f;
        double sum_rel_err = 0.0;

        for (int i = 0; i < matmul_M * matmul_N; i++) {
            float result = fp16_to_fp32(c_fp16[i]);
            float expected = bf16_to_fp32(c_bf16_ref[i]);
            float diff = fabsf(result - expected);
            float rel_err = (fabsf(expected) > 1e-3f) ? diff / fabsf(expected) : diff / 1e-3f;
            if (rel_err > 1.0f) rel_err = 1.0f;

            if (rel_err > max_rel_err) max_rel_err = rel_err;
            if (rel_err < min_rel_err) min_rel_err = rel_err;
            sum_rel_err += rel_err;

            if (i < 10) {
                printf(" -[%d] Result: %16.4f,  Expected: %16.4f,  RelErr: %4.4f%%\n", 
                       i, result, expected, rel_err * 100.0f);
            }
        }
        
        printf("\n Error Statistics (FP16 vs BF16):\n - Max: %4.4f%%\n - Min: %4.4f%%\n - Avg: %4.4f%%\n\n", 
               max_rel_err * 100.0f, min_rel_err * 100.0f, (float)(sum_rel_err / (matmul_M * matmul_N)) * 100.0f);

        stats_print(fp16_cycles);

        free(a_fp16);
        free(b_fp16);
        free(c_fp16);
        
    } else if (npu_type == NPU_TYPE_BF16) {
        // Use BF16 reference data directly
        uint16_t* c_bf16 = (uint16_t*)malloc(matmul_M * matmul_N * sizeof(uint16_t));
        if (!c_bf16) {
            printf("Failed to allocate BF16 output memory\n");
            return 1;
        }
        
        int bf16_cycles = 0;
        user_operator_matmul(&bf16_cycles, NPU_TYPE_BF16, matmul_M, matmul_N, matmul_K, a_bf16_ref, b_bf16_ref, c_bf16);

        printf("\nChecking first 10 results (BF16 result vs BF16 reference):\n");
        
        float max_rel_err = 0.0f;
        float min_rel_err = 1e9f;
        double sum_rel_err = 0.0;

        for (int i = 0; i < matmul_M * matmul_N; i++) {
            float result = bf16_to_fp32(c_bf16[i]);
            float expected = bf16_to_fp32(c_bf16_ref[i]);
            float diff = fabsf(result - expected);
            float rel_err = (fabsf(expected) > 1e-3f) ? diff / fabsf(expected) : diff / 1e-3f;
            if (rel_err > 1.0f) rel_err = 1.0f;

            if (rel_err > max_rel_err) max_rel_err = rel_err;
            if (rel_err < min_rel_err) min_rel_err = rel_err;
            sum_rel_err += rel_err;

            if (i < 10) {
                printf(" -[%d] Result: %16.4f,  Expected: %16.4f,  RelErr: %4.4f%%\n", 
                       i, result, expected, rel_err * 100.0f);
            }
        }
        
        printf("\n Error Statistics (BF16 vs BF16):\n - Max: %4.4f%%\n - Min: %4.4f%%\n - Avg: %4.4f%%\n\n", 
               max_rel_err * 100.0f, min_rel_err * 100.0f, (float)(sum_rel_err / (matmul_M * matmul_N)) * 100.0f);

        stats_print(bf16_cycles);

        free(c_bf16);
        
    } else if (npu_type == NPU_TYPE_FP8) {
        // Convert BF16 reference data to FP8 for operator input
        uint8_t* a_fp8 = (uint8_t*)malloc(matmul_M * matmul_K);
        uint8_t* b_fp8 = (uint8_t*)malloc(matmul_K * matmul_N);
        uint16_t* c_fp8_output_bf16 = (uint16_t*)malloc(matmul_M * matmul_N * sizeof(uint16_t));
        if (!a_fp8 || !b_fp8 || !c_fp8_output_bf16) {
            printf("Failed to allocate FP8 memory\n");
            return 1;
        }

        for(int i=0; i<matmul_M*matmul_K; i++) {
            a_fp8[i] = fp32_to_fp8_e4m3(bf16_to_fp32(a_bf16_ref[i]));
        }
        for(int i=0; i<matmul_K*matmul_N; i++) {
            b_fp8[i] = fp32_to_fp8_e4m3(bf16_to_fp32(b_bf16_ref[i]));
        }
        
        int fp8_cycles = 0;
        user_operator_matmul(&fp8_cycles, NPU_TYPE_FP8, matmul_M, matmul_N, matmul_K, a_fp8, b_fp8, c_fp8_output_bf16);

        printf("\nChecking first 10 results (FP8 result vs BF16 reference):\n");
        
        float max_rel_err = 0.0f;
        float min_rel_err = 1e9f;
        double sum_rel_err = 0.0;

        for (int i = 0; i < matmul_M * matmul_N; i++) {
            float result = bf16_to_fp32(c_fp8_output_bf16[i]);
            float expected = bf16_to_fp32(c_bf16_ref[i]);
            float diff = fabsf(result - expected);
            float rel_err = (fabsf(expected) > 1e-3f) ? diff / fabsf(expected) : diff / 1e-3f;
            if (rel_err > 1.0f) rel_err = 1.0f;

            if (rel_err > max_rel_err) max_rel_err = rel_err;
            if (rel_err < min_rel_err) min_rel_err = rel_err;
            sum_rel_err += rel_err;

            if (i < 10) {
                printf(" -[%d] Result: %16.4f,  Expected: %16.4f,  RelErr: %4.4f%%\n", 
                       i, result, expected, rel_err * 100.0f);
            }
        }
        
        printf("\n Error Statistics (FP8 vs BF16):\n - Max: %4.4f%%\n - Min: %4.4f%%\n - Avg: %4.4f%%\n\n", 
               max_rel_err * 100.0f, min_rel_err * 100.0f, (float)(sum_rel_err / (matmul_M * matmul_N)) * 100.0f);

        stats_print(fp8_cycles);

        free(a_fp8);
        free(b_fp8);
        free(c_fp8_output_bf16);
    }

    // Cleanup
    free(a_bf16_ref);
    free(b_bf16_ref);
    free(c_fp32_ref);
    free(c_bf16_ref);

    return 0;
}
