#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "npu_dma.h"
#include "npu_tpu.h"
#include "npu_rvv.h"
#include "operator_conv3d_patched.h"

// Conv3D Operator Implementation for Pre-Patched Input
// Input format: [num_patches, C_in, T, K_H, K_W] - already split into patches
// Implements Conv3D by decomposing into Conv2D pairs (via matmul)
// output = input_t0 @ weight_t0 + input_t1 @ weight_t1 + bias
// Output shape: [num_patches, C_out]
//
// Precision:
// - FP16 input: can use FP16 or FP32 accumulator (controlled by use_fp16_accu)
// - BF16 input: always uses FP32 accumulator
//
// Implementation:
// - D tensor is FP32 (or FP16 if use_fp16_accu=true for FP16)
// - First tile uses tpu_matmul_add(D = A*B + C) with C=bias
// - Subsequent tiles use tpu_matmul(D += A*B)
// - Final epilogue converts FP32 D back to original type

void user_operator_conv3d_patched(int* cycle_cnt, npu_datatype_t type,
                                   int num_patches,
                                   int C_in,
                                   int T,
                                   int K_H, int K_W,
                                   int C_out,
                                   void* input_ddr,
                                   void* weight_ddr,
                                   void* bias_ddr,
                                   void* rslt_ddr,
                                   bool use_fp16_accu) {
    
    if (type != NPU_TYPE_FP16 && type != NPU_TYPE_BF16) {
        fprintf(stderr, "ERROR: user_operator_conv3d_patched only supports NPU_TYPE_FP16 and NPU_TYPE_BF16\n");
        return;
    }
    
    // For BF16, use_fp16_accu must be false (FP32 accumulator is mandatory)
    if (type == NPU_TYPE_BF16 && use_fp16_accu) {
        fprintf(stderr, "ERROR: BF16 must use FP32 accumulator (use_fp16_accu must be false)\n");
        return;
    }
    
    if (T != 2) {
        fprintf(stderr, "ERROR: T must be 2 (T=%d)\n", T);
        return;
    }
    
    // Calculate dimensions
    int M = num_patches;                  // M dimension (num_patches)
    int K = C_in * K_H * K_W;             // K dimension (flattened spatial)
    int N = C_out;                        // N dimension
    
    // Determine accumulator type
    bool use_fp32_accu = !use_fp16_accu;  // FP32 unless explicitly using FP16
    npu_datatype_t accu_type = use_fp32_accu ? NPU_TYPE_FP32 : NPU_TYPE_FP16;
    
    printf("\nRunning User Operator: Conv3D Patched (ViT Patch Embed)\n");
    printf(" Input: [%d, %d, %d, %d, %d] (num_patches, C_in, T, K_H, K_W)\n", 
           num_patches, C_in, T, K_H, K_W);
    printf(" Kernel: [%d, %d, %d, %d, %d] (C_out, C_in, T, K_H, K_W)\n", 
           C_out, C_in, T, K_H, K_W);
    printf(" Output: [%d, %d] (num_patches, C_out)\n", M, N);
    printf(" MatMul: [%d, %d] x [%d, %d] x 2 frames + bias\n", M, K, K, N);
    printf(" Input type: %s, Accumulator: %s\n", 
           (type == NPU_TYPE_FP16) ? "FP16" : "BF16",
           use_fp32_accu ? "FP32" : "FP16");
    
    int thread_cycles[NPU_TPU_TOTAL_ARRAYS];
    for (int i = 0; i < NPU_TPU_TOTAL_ARRAYS; i++) thread_cycles[i] = 0;
    
    sram_reset();
    
    int type_size = npu_type_size(type);
    int accu_size = npu_type_size(accu_type);
    int tile_bytes_input = NPU_TPU_ARRAY_WIDTH * NPU_TPU_ARRAY_HEIGHT * type_size;
    int tile_bytes_accu = NPU_TPU_ARRAY_WIDTH * NPU_TPU_ARRAY_HEIGHT * accu_size;
    
    // Allocate SRAM for single tiles
    void* sram_a = sram_malloc(tile_bytes_input);
    void* sram_b = sram_malloc(tile_bytes_input);
    void* sram_c = sram_malloc(tile_bytes_accu);  // For bias tile (same type as D)
    void* sram_d = sram_malloc(tile_bytes_accu);  // D uses accumulator type
    
    if (!sram_a || !sram_b || !sram_c || !sram_d) {
        printf("OOM Conv3D Patched Tiles in SRAM\n");
        return;
    }
    
    int64_t dim_tile[] = {NPU_TPU_ARRAY_WIDTH, NPU_TPU_ARRAY_HEIGHT};
    
    // ========================================
    // Step 1: Reshape input and weights
    // Input: [num_patches, C_in, T, K_H, K_W] -> input_a[t]: [M, K]
    // Weight: [C_out, C_in, T, K_H, K_W] -> weight_b[t]: [K, N]
    // ========================================
    
    uint16_t* input_ptr = (uint16_t*)input_ddr;
    uint16_t* weight_ptr = (uint16_t*)weight_ddr;
    uint16_t* rslt_ptr = (uint16_t*)rslt_ddr;
    uint16_t* bias_ptr = (uint16_t*)bias_ddr;
    
    // Reshape input to input_a[t]: [M, K] where K = C_in * K_H * K_W
    uint16_t* input_a[2];
    int frame_spatial_size = K_H * K_W;
    int patch_size = C_in * T * frame_spatial_size;
    
    for (int t = 0; t < 2; t++) {
        input_a[t] = (uint16_t*)malloc(M * K * type_size);
        if (!input_a[t]) {
            printf("OOM input buffer for t=%d\n", t);
            return;
        }
        
        // Reshape: for each patch, extract frame t and flatten to [C_in * K_H * K_W]
        // Use im2col order: (h, w, c) - same as original conv3d operator
        for (int p = 0; p < M; p++) {
            for (int h = 0; h < K_H; h++) {
                for (int w = 0; w < K_W; w++) {
                    for (int c = 0; c < C_in; c++) {
                        // Source index in [num_patches, C_in, T, K_H, K_W]
                        int src_idx = p * patch_size + 
                                      c * (T * frame_spatial_size) + 
                                      t * frame_spatial_size + 
                                      h * K_W + w;
                        // Destination index in [M, K] where K = K_H * K_W * C_in
                        // im2col order: (h, w, c) = h * K_W * C_in + w * C_in + c
                        int dst_col = h * K_W * C_in + w * C_in + c;
                        int dst_idx = p * K + dst_col;
                        input_a[t][dst_idx] = input_ptr[src_idx];
                    }
                }
            }
        }
    }
    
    // Weight layout: [C_out, C_in, T, K_H, K_W]
    // Reshape to weight_b[t]: [K, N] where K = C_in * K_H * K_W, N = C_out
    uint16_t* weight_b[2];
    int weight_frame_size = frame_spatial_size;
    int weight_channel_size = T * weight_frame_size;
    int weight_out_size = C_in * weight_channel_size;
    
    for (int t = 0; t < 2; t++) {
        weight_b[t] = (uint16_t*)malloc(K * N * type_size);
        if (!weight_b[t]) {
            printf("OOM weight buffer for t=%d\n", t);
            return;
        }
        
        for (int oc = 0; oc < C_out; oc++) {
            for (int kh = 0; kh < K_H; kh++) {
                for (int kw = 0; kw < K_W; kw++) {
                    for (int ic = 0; ic < C_in; ic++) {
                        // Source index in [C_out, C_in, T, K_H, K_W]
                        int src_idx = oc * weight_out_size +
                                      ic * weight_channel_size +
                                      t * weight_frame_size +
                                      kh * K_W + kw;
                        // Destination index in [K, N]
                        // im2col order: (h, w, c) = kh * K_W * C_in + kw * C_in + ic
                        int row = kh * K_W * C_in + kw * C_in + ic;
                        int dst_idx = row * N + oc;
                        weight_b[t][dst_idx] = weight_ptr[src_idx];
                    }
                }
            }
        }
    }
    
    // ========================================
    // Step 2: Prepare bias in accumulator format (FP32 or FP16)
    // ========================================
    
    void* bias_accu = malloc(N * accu_size);
    if (!bias_accu) {
        printf("OOM bias buffer\n");
        for (int t = 0; t < 2; t++) { free(input_a[t]); free(weight_b[t]); }
        return;
    }
    
    if (use_fp32_accu) {
        float* bias_fp32 = (float*)bias_accu;
        for (int n = 0; n < N; n++) {
            if (bias_ptr) {
                bias_fp32[n] = (type == NPU_TYPE_FP16) ? 
                    fp16_to_fp32(bias_ptr[n]) : bf16_to_fp32(bias_ptr[n]);
            } else {
                bias_fp32[n] = 0.0f;
            }
        }
    } else {
        uint16_t* bias_fp16 = (uint16_t*)bias_accu;
        for (int n = 0; n < N; n++) {
            if (bias_ptr) {
                bias_fp16[n] = bias_ptr[n];
            } else {
                bias_fp16[n] = fp32_to_fp16(0.0f);
            }
        }
    }
    
    // ========================================
    // Step 3: Allocate output buffer in accumulator format
    // ========================================
    
    void* rslt_accu = malloc(M * N * accu_size);
    if (!rslt_accu) {
        printf("OOM output buffer\n");
        free(bias_accu);
        for (int t = 0; t < 2; t++) { free(input_a[t]); free(weight_b[t]); }
        return;
    }
    memset(rslt_accu, 0, M * N * accu_size);
    
    // ========================================
    // Step 4: Tiled MatMul for both frames
    // D = input_a[0] × weight_b[0] + input_a[1] × weight_b[1] + bias
    // First K tile: D = A * B + bias (tpu_matmul_add)
    // Subsequent K tiles: D += A * B (tpu_matmul)
    // ========================================
    
    int tile_M = M / NPU_TPU_ARRAY_HEIGHT;
    int tile_N = N / NPU_TPU_ARRAY_WIDTH;
    int tile_K = K / NPU_TPU_ARRAY_WIDTH;
    
    // Handle non-divisible dimensions
    if (M % NPU_TPU_ARRAY_HEIGHT != 0) tile_M++;
    if (N % NPU_TPU_ARRAY_WIDTH != 0) tile_N++;
    if (K % NPU_TPU_ARRAY_WIDTH != 0) tile_K++;
    
    printf(" Tiling: tile_M=%d, tile_N=%d, tile_K=%d\n", tile_M, tile_N, tile_K);
    
    for (int cluster_id = 0; cluster_id < NPU_NUM_CLUSTERS; cluster_id++) {
        for (int array_id = 0; array_id < NPU_TPU_ARRAYS_PER_CLUSTER; array_id++) {
            int global_tid = cluster_id * NPU_TPU_ARRAYS_PER_CLUSTER + array_id;
            int* current_cycles = &thread_cycles[global_tid];
            int total_tiles = tile_M * tile_N;
            
            for (int task_idx = global_tid; task_idx < total_tiles; task_idx += NPU_TPU_TOTAL_ARRAYS) {
                int m_idx = task_idx / tile_N;
                int n_idx = task_idx % tile_N;
                
                // Calculate actual tile dimensions (handle boundary)
                int m_start = m_idx * NPU_TPU_ARRAY_HEIGHT;
                int n_start = n_idx * NPU_TPU_ARRAY_WIDTH;
                int actual_tile_m = (m_start + NPU_TPU_ARRAY_HEIGHT <= M) ? NPU_TPU_ARRAY_HEIGHT : (M - m_start);
                int actual_tile_n = (n_start + NPU_TPU_ARRAY_WIDTH <= N) ? NPU_TPU_ARRAY_WIDTH : (N - n_start);
                
                if (actual_tile_m <= 0 || actual_tile_n <= 0) continue;
                
                // Prepare bias tile in SRAM (C for tpu_matmul_add)
                // Bias is broadcast across M dimension
                npu_tensor_t t_c_sram = create_tensor(sram_c, 2, dim_tile, accu_type);
                if (use_fp32_accu) {
                    float* c_sram = (float*)sram_c;
                    float* bias_fp32 = (float*)bias_accu;
                    for (int m = 0; m < NPU_TPU_ARRAY_HEIGHT; m++) {
                        for (int n = 0; n < NPU_TPU_ARRAY_WIDTH; n++) {
                            int bias_idx = n_start + n;
                            c_sram[m * NPU_TPU_ARRAY_WIDTH + n] = (bias_idx < N) ? bias_fp32[bias_idx] : 0.0f;
                        }
                    }
                } else {
                    uint16_t* c_sram = (uint16_t*)sram_c;
                    uint16_t* bias_fp16 = (uint16_t*)bias_accu;
                    for (int m = 0; m < NPU_TPU_ARRAY_HEIGHT; m++) {
                        for (int n = 0; n < NPU_TPU_ARRAY_WIDTH; n++) {
                            int bias_idx = n_start + n;
                            c_sram[m * NPU_TPU_ARRAY_WIDTH + n] = (bias_idx < N) ? bias_fp16[bias_idx] : fp32_to_fp16(0.0f);
                        }
                    }
                }
                
                // D tile setup
                npu_tensor_t t_d_sram = create_tensor(sram_d, 2, dim_tile, accu_type);
                
                bool first_k_tile = true;
                
                // Process both frames
                for (int t = 0; t < 2; t++) {
                    // K loop
                    for (int k_idx = 0; k_idx < tile_K; k_idx++) {
                        int k_start = k_idx * NPU_TPU_ARRAY_WIDTH;
                        int actual_tile_k = (k_start + NPU_TPU_ARRAY_WIDTH <= K) ? NPU_TPU_ARRAY_WIDTH : (K - k_start);
                        
                        if (actual_tile_k <= 0) continue;
                        
                        // Load A tile
                        int64_t offset_a = ((int64_t)m_start * K) + k_start;
                        uint16_t* a_ptr = input_a[t] + offset_a;
                        
                        npu_tensor_t t_a_ddr = create_tensor(a_ptr, 2, dim_tile, type);
                        t_a_ddr.nb[1] = K * type_size;
                        npu_tensor_t t_a_sram = create_tensor(sram_a, 2, dim_tile, type);
                        dma_copy(&t_a_sram, &t_a_ddr, current_cycles);
                        
                        // Load B tile
                        int64_t offset_b = ((int64_t)k_start * N) + n_start;
                        uint16_t* b_ptr = weight_b[t] + offset_b;
                        
                        npu_tensor_t t_b_ddr = create_tensor(b_ptr, 2, dim_tile, type);
                        t_b_ddr.nb[1] = N * type_size;
                        npu_tensor_t t_b_sram = create_tensor(sram_b, 2, dim_tile, type);
                        dma_copy(&t_b_sram, &t_b_ddr, current_cycles);
                        
                        if (first_k_tile) {
                            // First K tile: D = A * B + bias
                            tpu_matmul_add(&t_d_sram, &t_a_sram, &t_b_sram, &t_c_sram, 
                                          current_cycles, type, use_fp16_accu);
                            first_k_tile = false;
                        } else {
                            // Subsequent K tiles: D += A * B
                            tpu_matmul(&t_d_sram, &t_a_sram, &t_b_sram, 
                                      current_cycles, type, use_fp16_accu);
                        }
                    }
                }
                
                // Store D tile to output buffer
                int64_t offset_d = ((int64_t)m_start * N) + n_start;
                if (use_fp32_accu) {
                    float* d_ptr_accu = (float*)rslt_accu + offset_d;
                    npu_tensor_t t_d_ddr = create_tensor(d_ptr_accu, 2, dim_tile, accu_type);
                    t_d_ddr.nb[1] = N * accu_size;
                    dma_copy(&t_d_ddr, &t_d_sram, current_cycles);
                } else {
                    uint16_t* d_ptr_accu = (uint16_t*)rslt_accu + offset_d;
                    npu_tensor_t t_d_ddr = create_tensor(d_ptr_accu, 2, dim_tile, accu_type);
                    t_d_ddr.nb[1] = N * accu_size;
                    dma_copy(&t_d_ddr, &t_d_sram, current_cycles);
                }
            }
        }
    }
    
    // ========================================
    // Step 5: Convert accumulator result back to original type (epilogue)
    // ========================================
    
    if (use_fp32_accu) {
        float* rslt_fp32 = (float*)rslt_accu;
        for (int i = 0; i < M * N; i++) {
            rslt_ptr[i] = (type == NPU_TYPE_FP16) ? 
                fp32_to_fp16(rslt_fp32[i]) : fp32_to_bf16(rslt_fp32[i]);
        }
    } else {
        // FP16 accumulator -> FP16 output (direct copy)
        memcpy(rslt_ptr, rslt_accu, M * N * type_size);
    }
    
    // Cleanup
    free(rslt_accu);
    free(bias_accu);
    for (int t = 0; t < 2; t++) {
        free(input_a[t]);
        free(weight_b[t]);
    }
    
    // Calculate max cycles
    int max_cycles = 0;
    for (int i = 0; i < NPU_TPU_TOTAL_ARRAYS; i++) {
        if (thread_cycles[i] > max_cycles) max_cycles = thread_cycles[i];
    }
    *cycle_cnt += max_cycles;
}
