#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "npu_common.h"
#include "npu_sram.h"
#include "operator/operator_silu.h"
#include "t000_silu.h"

// Test function for SILU operator
int test_silu(int rows, int cols) {
    printf(" Case: SILU TEST (FP32)\n");
    printf("==============================================================\n");

    stats_reset();
    sram_init();
    // Init done
    
    int total_cycles = 0;

    int silu_Rows = rows;
    int silu_Cols = cols;
    
    // Data is in DDR (Host Memory) - too large for SRAM
    float* data_ddr = (float*)malloc(silu_Rows * silu_Cols * sizeof(float));
    if (!data_ddr) { printf("Failed to allocate DDR memory\n"); return 1; }
    // Result is in DDR (Host Memory) - too large for SRAM
    float* rslt_ddr = (float*)malloc(silu_Rows * silu_Cols * sizeof(float));
    if (!rslt_ddr) { printf("Failed to allocate DDR memory\n"); return 1; }
    
    // Generate random values for SILU test
    unsigned int seed = time(NULL);
    float silu_max = 10.0f;
    float silu_min = -10.0f;
    for(int i=0; i<silu_Rows*silu_Cols; i++) {
        float rand_val = silu_min + ((float)(rand_r(&seed)) / (float)RAND_MAX) * (silu_max - silu_min);
        data_ddr[i] = rand_val;
    }

    user_operator_silu(&total_cycles, silu_Rows, silu_Cols, data_ddr, rslt_ddr);
    
    {
        printf("\nChecking first 10 results:\n");
        
        float max_abs_err = 0.0f;
        float min_abs_err = 1e9f;
        double sum_abs_err = 0.0;
        float max_rel_err = 0.0f;
        float min_rel_err = 1e9f;
        double sum_rel_err = 0.0;
        
        for (int i = 0; i < silu_Rows * silu_Cols; i++) {
            float input = data_ddr[i];
            float result = rslt_ddr[i];
            float exp_neg_x = expf(-input);
            float expected = input / (1.0f + exp_neg_x);
            
            // Calculate absolute error
            float abs_err = fabsf(result - expected);
            
            // Calculate relative error
            float abs_expected = fabsf(expected);
            float rel_err = (abs_expected > 0.0f) ? abs_err / abs_expected : 0.0f;
            
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (abs_err < min_abs_err) min_abs_err = abs_err;
            sum_abs_err += abs_err;
            
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            if (rel_err < min_rel_err) min_rel_err = rel_err;
            sum_rel_err += rel_err;
            
            if (i < 10) {
                printf(" -[%d] Input: %10.6f,  Result: %10.6f,  Expected: %10.6f,  AbsErr: %10.6f,  RelErr: %10.6f%%\n", 
                       i, input, result, expected, abs_err, rel_err * 100.0f);
            }
        }
        
        printf("\n Error Statistics:\n");
        printf(" Absolute Error:\n - Max: %10.6f\n - Min: %10.6f\n - Avg: %10.6f\n", 
               max_abs_err, min_abs_err, (float)(sum_abs_err / (silu_Rows * silu_Cols)));
        printf(" Relative Error:\n - Max: %10.6f%%\n - Min: %10.6f%%\n - Avg: %10.6f%%\n\n", 
               max_rel_err * 100.0f, min_rel_err * 100.0f, (float)(sum_rel_err / (silu_Rows * silu_Cols)) * 100.0f);
    }
    
    stats_print(total_cycles);

    free(rslt_ddr);
    free(data_ddr);
    
    return 0;
}

