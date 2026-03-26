#ifndef T003_CONV3D_H
#define T003_CONV3D_H

#ifdef __cplusplus
extern "C" {
#endif

// Test function for Conv3D operator (ViT Patch Embed)
// Default case: Qwen3-VL-4B ViT with 448x448x3 image input
// Parameters:
//   H, W   - Input image dimensions (default: 448x448)
//   C_in   - Input channels (default: 3 for RGB)
//   T      - Temporal dimension (default: 2)
//   K_H, K_W - Kernel spatial dimensions (default: 16x16)
//   C_out  - Output channels (default: 1024)
//   type   - Data type string: "fp16" or "bf16"
int test_conv3d(int H, int W, int C_in, int T, int K_H, int K_W, int C_out, const char* type);

#ifdef __cplusplus
}
#endif

#endif // T003_CONV3D_H

