#ifndef T004_CONV3D_PATCHED_H
#define T004_CONV3D_PATCHED_H

#ifdef __cplusplus
extern "C" {
#endif

// Test function for Conv3D operator with pre-patched input (ViT Patch Embed)
// Loads data from files in the data/ directory:
//   - conv3d_input_bf16.txt:  Input tensor [num_patches, C_in, T, K_H, K_W]
//   - conv3d_weight_bf16.txt: Weight tensor [C_out, C_in, T, K_H, K_W]
//   - conv3d_bias_bf16.txt:   Bias tensor [C_out]
//   - conv3d_output_bf16.txt: Expected output for comparison [num_patches, C_out]
//
// Data format: BF16 values packed as 2xBF16 per UINT32 (little-endian)
// Precision: BF16 input with FP32 accumulator (mandatory)
//
// Parameters:
//   data_dir - Path to the data directory containing the input files
//
// Returns:
//   0 on success, non-zero on failure
int test_conv3d_patched(const char* data_dir);

#ifdef __cplusplus
}
#endif

#endif // T004_CONV3D_PATCHED_H
