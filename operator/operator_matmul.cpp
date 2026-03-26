#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "npu_dma.h"
#include "npu_tpu.h"
#include "operator_matmul.h"

// MatMul Operator
// D = A * B
// Precision:
// - FP16 input: FP32 accumulator (default), can use FP16 with use_fp16_accu=true
// - BF16 input: FP32 accumulator (mandatory)
// - FP8 input: FP32 accumulator (mandatory)
void user_operator_matmul(int* cycle_cnt, npu_datatype_t type, int M, int N, int K, 
                          void* a_ddr, void* b_ddr, void* rslt_ddr) {
    if (type != NPU_TYPE_FP16 && type != NPU_TYPE_BF16 && type != NPU_TYPE_FP8) {
        fprintf(stderr, "ERROR: user_operator_matmul only supports NPU_TYPE_FP16, NPU_TYPE_BF16, and NPU_TYPE_FP8\n");
        return;
    }
    
    printf("\nRunning User Operator: MatMul [%d, %d] x [%d, %d]...\n", M, K, K, N);
    printf(" Input type: %s, Accumulator: FP32\n", 
           (type == NPU_TYPE_FP16) ? "FP16" : (type == NPU_TYPE_BF16) ? "BF16" : "FP8");
    
    int thread_cycles[NPU_TPU_TOTAL_ARRAYS];
    for(int i=0; i<NPU_TPU_TOTAL_ARRAYS; i++) thread_cycles[i] = 0;

    sram_reset();
    
    npu_datatype_t input_type = type;
    npu_datatype_t output_type = (type == NPU_TYPE_FP8) ? NPU_TYPE_BF16 : type;
    npu_datatype_t accu_type = NPU_TYPE_FP32;  // Always FP32 accumulator
    
    int tile_bytes_input = NPU_TPU_ARRAY_WIDTH * NPU_TPU_ARRAY_HEIGHT * npu_type_size(input_type);
    int tile_bytes_accu = NPU_TPU_ARRAY_WIDTH * NPU_TPU_ARRAY_HEIGHT * sizeof(float);
    
    void* sram_a = sram_malloc(tile_bytes_input);
    void* sram_b = sram_malloc(tile_bytes_input);
    void* sram_d = sram_malloc(tile_bytes_accu);  // D uses FP32

    if (!sram_a || !sram_b || !sram_d) {
        printf("OOM MatMul Tiles in SRAM\n"); 
        return; 
    }

    int64_t dim_tile[] = {NPU_TPU_ARRAY_WIDTH, NPU_TPU_ARRAY_HEIGHT};
    
    // Allocate FP32 output buffer
    float* rslt_fp32 = (float*)malloc(M * N * sizeof(float));
    if (!rslt_fp32) {
        printf("OOM FP32 output buffer\n");
        return;
    }
    memset(rslt_fp32, 0, M * N * sizeof(float));

    for (int cluster_id = 0; cluster_id < NPU_NUM_CLUSTERS; cluster_id++) {
        for (int array_id = 0; array_id < NPU_TPU_ARRAYS_PER_CLUSTER; array_id++) {
            int global_tid = cluster_id * NPU_TPU_ARRAYS_PER_CLUSTER + array_id;
            int* current_cycles = &thread_cycles[global_tid];
            int total_tiles = (M * N) / (NPU_TPU_ARRAY_WIDTH * NPU_TPU_ARRAY_HEIGHT);
            
            for (int task_idx = global_tid; task_idx < total_tiles; task_idx += NPU_TPU_TOTAL_ARRAYS) {
                int m_idx = task_idx / (N / NPU_TPU_ARRAY_WIDTH);
                int n_idx = task_idx % (N / NPU_TPU_ARRAY_WIDTH);

                // Initialize D tile to zero in SRAM (FP32)
                memset(sram_d, 0, tile_bytes_accu);
                npu_tensor_t t_d_sram = create_tensor(sram_d, 2, dim_tile, accu_type);

                for (int k_idx = 0; k_idx < K / NPU_TPU_ARRAY_WIDTH; k_idx++) {
                    int64_t offset_a = ((int64_t)m_idx * NPU_TPU_ARRAY_HEIGHT * K) + (k_idx * NPU_TPU_ARRAY_WIDTH);
                    void* a_src_ptr = NULL;
                    if (type == NPU_TYPE_FP16) a_src_ptr = ((uint16_t*)a_ddr) + offset_a;
                    else if (type == NPU_TYPE_BF16) a_src_ptr = ((uint16_t*)a_ddr) + offset_a;
                    else if (type == NPU_TYPE_FP8) a_src_ptr = ((uint8_t*)a_ddr) + offset_a;
                    else a_src_ptr = (char*)a_ddr + offset_a * npu_type_size(type);

                    npu_tensor_t t_a_ddr = create_tensor(a_src_ptr, 2, dim_tile, input_type);
                    t_a_ddr.nb[1] = K * npu_type_size(input_type);
                    npu_tensor_t t_a_sram = create_tensor(sram_a, 2, dim_tile, input_type);
                    dma_copy(&t_a_sram, &t_a_ddr, current_cycles);

                    int64_t offset_b = ((int64_t)k_idx * NPU_TPU_ARRAY_HEIGHT * N) + (n_idx * NPU_TPU_ARRAY_WIDTH);
                    void* b_src_ptr = NULL;
                    if (type == NPU_TYPE_FP16) b_src_ptr = ((uint16_t*)b_ddr) + offset_b;
                    else if (type == NPU_TYPE_BF16) b_src_ptr = ((uint16_t*)b_ddr) + offset_b;
                    else if (type == NPU_TYPE_FP8) b_src_ptr = ((uint8_t*)b_ddr) + offset_b;
                    else b_src_ptr = (char*)b_ddr + offset_b * npu_type_size(type);

                    npu_tensor_t t_b_ddr = create_tensor(b_src_ptr, 2, dim_tile, input_type);
                    t_b_ddr.nb[1] = N * npu_type_size(input_type);
                    npu_tensor_t t_b_sram = create_tensor(sram_b, 2, dim_tile, input_type);
                    dma_copy(&t_b_sram, &t_b_ddr, current_cycles);

                    // D += A * B (FP32 accumulator)
                    tpu_matmul(&t_d_sram, &t_a_sram, &t_b_sram, current_cycles, input_type, false);
                }

                // Store D tile to FP32 output buffer
                int64_t offset_d = ((int64_t)m_idx * NPU_TPU_ARRAY_HEIGHT * N) + (n_idx * NPU_TPU_ARRAY_WIDTH);
                float* d_dst_ptr = rslt_fp32 + offset_d;

                npu_tensor_t t_d_ddr = create_tensor(d_dst_ptr, 2, dim_tile, accu_type);
                t_d_ddr.nb[1] = N * sizeof(float);
                dma_copy(&t_d_ddr, &t_d_sram, current_cycles);
            }
        }
    }
    
    // Epilogue: Convert FP32 result back to output type
    if (output_type == NPU_TYPE_FP16) {
        uint16_t* output_ptr = (uint16_t*)rslt_ddr;
        for (int i = 0; i < M * N; i++) {
            output_ptr[i] = fp32_to_fp16(rslt_fp32[i]);
        }
    } else if (output_type == NPU_TYPE_BF16) {
        uint16_t* output_ptr = (uint16_t*)rslt_ddr;
        for (int i = 0; i < M * N; i++) {
            output_ptr[i] = fp32_to_bf16(rslt_fp32[i]);
        }
    }
    
    free(rslt_fp32);

    // Calculate max cycles among all arrays
    int max_cycles = 0;
    for(int i=0; i<NPU_TPU_TOTAL_ARRAYS; i++) {
        if(thread_cycles[i] > max_cycles) max_cycles = thread_cycles[i];
    }
    *cycle_cnt += max_cycles;
}
