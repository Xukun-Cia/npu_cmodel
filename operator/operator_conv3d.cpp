#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "npu_dma.h"
#include "npu_tpu.h"
#include "npu_rvv.h"
#include "operator_conv3d.h"

// Conv3D Operator Implementation
// Implements Conv3D by decomposing into Conv2D pairs (via im2col->matmul) 
// Every 2 frames produce one output: output = conv2d(frame_t) + conv2d(frame_t+1) + bias
// Output shape: [(T/2) * num_patches, C_out]
// Precision:
// - FP16 input: FP32 accumulator (default)
// - BF16 input: FP32 accumulator (mandatory)
// Implementation notes:
// - Uses per-tile DMA with stride support for correctness
// - sram_load/store have 0 cycle cost (pipelined with matmul)
// - D tensor uses FP32, converted back to original type in epilogue

void user_operator_conv3d(int* cycle_cnt, npu_datatype_t type,
                          int H, int W, int C_in,
                          int T,
                          int K_H, int K_W,
                          int C_out,
                          void* input_ddr,
                          void* weight_ddr,
                          void* bias_ddr,
                          void* rslt_ddr) {
    
    if (type != NPU_TYPE_FP16 && type != NPU_TYPE_BF16) {
        fprintf(stderr, "ERROR: user_operator_conv3d only supports NPU_TYPE_FP16 and NPU_TYPE_BF16\n");
        return;
    }
    
    if (T % 2 != 0) {
        fprintf(stderr, "ERROR: T must be even (T=%d)\n", T);
        return;
    }
    
    // Calculate dimensions
    int num_patches_h = H / K_H;
    int num_patches_w = W / K_W;
    int M = num_patches_h * num_patches_w;  // M dimension (num_patches)
    int K = K_H * K_W * C_in;               // K dimension (kernel_elements)
    int N = C_out;                           // N dimension
    int num_frame_pairs = T / 2;
    int total_output_patches = num_frame_pairs * M;
    
    printf("\nRunning User Operator: Conv3D (ViT Patch Embed)\n");
    printf(" Input: [%d, %d, %d, %d] (T, H, W, C)\n", T, H, W, C_in);
    printf(" Kernel: [%d, %d, 2, %d, %d] (C_out, C_in, T_kernel=2, K_H, K_W)\n", C_out, C_in, K_H, K_W);
    printf(" Output: [%d, %d] (total_patches = %d frame_pairs * %d patches, C_out)\n", 
           total_output_patches, C_out, num_frame_pairs, M);
    printf(" Each frame pair: matmul [%d, %d] x [%d, %d] x 2 + bias\n", M, K, K, N);
    printf(" Input type: %s, Accumulator: FP32\n", (type == NPU_TYPE_FP16) ? "FP16" : "BF16");
    
    int thread_cycles[NPU_TPU_TOTAL_ARRAYS];
    for (int i = 0; i < NPU_TPU_TOTAL_ARRAYS; i++) thread_cycles[i] = 0;
    
    sram_reset();
    
    int type_size = npu_type_size(type);
    int tile_bytes_input = NPU_TPU_ARRAY_WIDTH * NPU_TPU_ARRAY_HEIGHT * type_size;
    int tile_bytes_accu = NPU_TPU_ARRAY_WIDTH * NPU_TPU_ARRAY_HEIGHT * sizeof(float);
    
    // Allocate SRAM for single tiles (D uses FP32)
    void* sram_a = sram_malloc(tile_bytes_input);
    void* sram_b = sram_malloc(tile_bytes_input);
    void* sram_d = sram_malloc(tile_bytes_accu);
    
    if (!sram_a || !sram_b || !sram_d) {
        printf("OOM Conv3D Tiles in SRAM\n");
        return;
    }
    
    int64_t dim_tile[] = {NPU_TPU_ARRAY_WIDTH, NPU_TPU_ARRAY_HEIGHT};
    
    // ========================================
    // Step 1: Reshape weights (CPU preprocessing)
    // Weight: [C_out, C_in, T=2, K_H, K_W] -> weight_b[t]: [K, N]
    // ========================================
    
    uint16_t* weight_ptr = (uint16_t*)weight_ddr;
    uint16_t* weight_b[2];
    
    for (int t = 0; t < 2; t++) {
        weight_b[t] = (uint16_t*)malloc(K * N * type_size);
        if (!weight_b[t]) {
            printf("OOM weight buffer for t=%d\n", t);
            return;
        }
        
        for (int oc = 0; oc < C_out; oc++) {
            for (int ic = 0; ic < C_in; ic++) {
                for (int kh = 0; kh < K_H; kh++) {
                    for (int kw = 0; kw < K_W; kw++) {
                        int src_idx = oc * (C_in * 2 * K_H * K_W) +
                                      ic * (2 * K_H * K_W) +
                                      t * (K_H * K_W) +
                                      kh * K_W + kw;
                        int row = kh * K_W * C_in + kw * C_in + ic;  // K dimension
                        int dst_idx = row * N + oc;  // [K, N] row-major
                        weight_b[t][dst_idx] = weight_ptr[src_idx];
                    }
                }
            }
        }
    }
    
    uint16_t* input_ptr = (uint16_t*)input_ddr;
    uint16_t* rslt_ptr = (uint16_t*)rslt_ddr;
    uint16_t* bias_ptr = (uint16_t*)bias_ddr;
    int input_frame_size = H * W * C_in;
    
    // ========================================
    // Step 2: Process each frame pair
    // ========================================
    
    for (int fp = 0; fp < num_frame_pairs; fp++) {
        int frame_0 = fp * 2;
        int frame_1 = fp * 2 + 1;
        int output_row_offset = fp * M;
        
        // im2col for both frames: [M, K]
        uint16_t* im2col[2];
        for (int t = 0; t < 2; t++) {
            im2col[t] = (uint16_t*)malloc(M * K * type_size);
            if (!im2col[t]) {
                printf("OOM im2col buffer\n");
                return;
            }
            
            int frame_idx = (t == 0) ? frame_0 : frame_1;
            uint16_t* frame_ptr = input_ptr + frame_idx * input_frame_size;
            
            for (int ph = 0; ph < num_patches_h; ph++) {
                for (int pw = 0; pw < num_patches_w; pw++) {
                    int patch_idx = ph * num_patches_w + pw;
                    int src_h_start = ph * K_H;
                    int src_w_start = pw * K_W;
                    
                    int col = 0;
                    for (int kh = 0; kh < K_H; kh++) {
                        for (int kw = 0; kw < K_W; kw++) {
                            for (int c = 0; c < C_in; c++) {
                                int src_h = src_h_start + kh;
                                int src_w = src_w_start + kw;
                                int src_idx = src_h * W * C_in + src_w * C_in + c;
                                im2col[t][patch_idx * K + col] = frame_ptr[src_idx];
                                col++;
                            }
                        }
                    }
                }
            }
        }
        
        // Allocate FP32 output buffer for this frame pair
        float* rslt_fp32 = (float*)malloc(M * N * sizeof(float));
        if (!rslt_fp32) {
            printf("OOM FP32 output buffer\n");
            for (int t = 0; t < 2; t++) free(im2col[t]);
            return;
        }
        
        // Initialize with bias (converted to FP32)
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int out_idx = m * N + n;
                if (bias_ptr) {
                    rslt_fp32[out_idx] = (type == NPU_TYPE_FP16) ? 
                        fp16_to_fp32(bias_ptr[n]) : bf16_to_fp32(bias_ptr[n]);
                } else {
                    rslt_fp32[out_idx] = 0.0f;
                }
            }
        }
        
        // ========================================
        // Tiled MatMul: D = im2col[0] × weight_b[0] + im2col[1] × weight_b[1] + bias
        // ========================================
        
        int tile_M = M / NPU_TPU_ARRAY_HEIGHT;
        int tile_N = N / NPU_TPU_ARRAY_WIDTH;
        int tile_K = K / NPU_TPU_ARRAY_WIDTH;
        
        for (int t = 0; t < 2; t++) {
            for (int cluster_id = 0; cluster_id < NPU_NUM_CLUSTERS; cluster_id++) {
                for (int array_id = 0; array_id < NPU_TPU_ARRAYS_PER_CLUSTER; array_id++) {
                    int global_tid = cluster_id * NPU_TPU_ARRAYS_PER_CLUSTER + array_id;
                    int* current_cycles = &thread_cycles[global_tid];
                    int total_tiles = tile_M * tile_N;
                    
                    for (int task_idx = global_tid; task_idx < total_tiles; task_idx += NPU_TPU_TOTAL_ARRAYS) {
                        int m_idx = task_idx / tile_N;
                        int n_idx = task_idx % tile_N;
                        
                        // Load D tile from FP32 buffer to SRAM
                        int64_t offset_d = ((int64_t)m_idx * NPU_TPU_ARRAY_HEIGHT * N) + 
                                           (n_idx * NPU_TPU_ARRAY_WIDTH);
                        float* d_ptr = rslt_fp32 + offset_d;
                        
                        npu_tensor_t t_d_ddr = create_tensor(d_ptr, 2, dim_tile, NPU_TYPE_FP32);
                        t_d_ddr.nb[1] = N * sizeof(float);
                        npu_tensor_t t_d_sram = create_tensor(sram_d, 2, dim_tile, NPU_TYPE_FP32);
                        dma_copy(&t_d_sram, &t_d_ddr, current_cycles);
                        
                        // K loop: D += A × B
                        for (int k_idx = 0; k_idx < tile_K; k_idx++) {
                            // Load A tile
                            int64_t offset_a = ((int64_t)m_idx * NPU_TPU_ARRAY_HEIGHT * K) + 
                                               (k_idx * NPU_TPU_ARRAY_WIDTH);
                            uint16_t* a_ptr = im2col[t] + offset_a;
                            
                            npu_tensor_t t_a_ddr = create_tensor(a_ptr, 2, dim_tile, type);
                            t_a_ddr.nb[1] = K * type_size;
                            npu_tensor_t t_a_sram = create_tensor(sram_a, 2, dim_tile, type);
                            dma_copy(&t_a_sram, &t_a_ddr, current_cycles);
                            
                            // Load B tile
                            int64_t offset_b = ((int64_t)k_idx * NPU_TPU_ARRAY_HEIGHT * N) + 
                                               (n_idx * NPU_TPU_ARRAY_WIDTH);
                            uint16_t* b_ptr = weight_b[t] + offset_b;
                            
                            npu_tensor_t t_b_ddr = create_tensor(b_ptr, 2, dim_tile, type);
                            t_b_ddr.nb[1] = N * type_size;
                            npu_tensor_t t_b_sram = create_tensor(sram_b, 2, dim_tile, type);
                            dma_copy(&t_b_sram, &t_b_ddr, current_cycles);
                            
                            // TPU matmul: D += A × B (FP32 accumulator)
                            tpu_matmul(&t_d_sram, &t_a_sram, &t_b_sram, current_cycles, type, false);
                        }
                        
                        // Store D tile back to FP32 buffer
                        dma_copy(&t_d_ddr, &t_d_sram, current_cycles);
                    }
                }
            }
        }
        
        // Epilogue: Convert FP32 result back to original type
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int out_idx = (output_row_offset + m) * N + n;
                int fp32_idx = m * N + n;
                if (type == NPU_TYPE_FP16) {
                    rslt_ptr[out_idx] = fp32_to_fp16(rslt_fp32[fp32_idx]);
                } else {
                    rslt_ptr[out_idx] = fp32_to_bf16(rslt_fp32[fp32_idx]);
                }
            }
        }
        
        // Cleanup
        free(rslt_fp32);
        for (int t = 0; t < 2; t++) {
            free(im2col[t]);
        }
    }
    
    // Cleanup
    for (int t = 0; t < 2; t++) {
        free(weight_b[t]);
    }
    
    // Calculate max cycles
    int max_cycles = 0;
    for (int i = 0; i < NPU_TPU_TOTAL_ARRAYS; i++) {
        if (thread_cycles[i] > max_cycles) max_cycles = thread_cycles[i];
    }
    *cycle_cnt += max_cycles;
}
