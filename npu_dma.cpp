#include "npu_dma.h"
#include <string.h>
#include <stdio.h>

void dma_copy(npu_tensor_t* dst, const npu_tensor_t* src, int* t) {
    // Validate basic compatibility
    int64_t src_ne = npu_tensor_ne(src);
    // int64_t dst_ne = npu_tensor_ne(dst); // Unused
    int type_size = npu_type_size(src->type);
    
    // Total bytes to copy
    size_t total_bytes = src_ne * type_size;

    // Perform Copy
    if (src->data && dst->data) {
        // Check if contiguous copy is possible
        // Simple heuristic: if strides match default dense packing for both, use memcpy.
        // For now, we just check if ndim >= 2 and strides look non-contiguous.
        bool is_strided = false;
        if (src->ndim >= 2) {
            if (src->nb[1] != (size_t)(src->ne[0] * type_size)) is_strided = true;
        }
        if (dst->ndim >= 2) {
             if (dst->nb[1] != (size_t)(dst->ne[0] * type_size)) is_strided = true;
        }

        if (!is_strided) {
            memcpy(dst->data, src->data, total_bytes);
        } else {
            // Handle 2D strided copy (common case for tiles)
            // Assume ndim=2 for tiles
            int rows = (src->ndim >= 2) ? src->ne[1] : 1;
            int cols = src->ne[0]; // Elements per row
            size_t row_bytes = cols * type_size;
            
            char* src_ptr = (char*)src->data;
            char* dst_ptr = (char*)dst->data;
            
            for (int i = 0; i < rows; i++) {
                memcpy(dst_ptr, src_ptr, row_bytes);
                if (src->ndim >= 2) src_ptr += src->nb[1];
                else src_ptr += row_bytes;
                
                if (dst->ndim >= 2) dst_ptr += dst->nb[1];
                else dst_ptr += row_bytes;
            }
        }
    }
    
    // Calculate cycles
    // Bandwidth: NPU_DMA_BANDWIDTH_BYTES bytes/cycle
    int cycles = 10 + (total_bytes / NPU_DMA_BANDWIDTH_BYTES); 
    
    if (t) *t += cycles;
    //printf("DMA copy: %d cycles\n", cycles);
    // Update stats
    g_npu_stats.inst_dma++;
}
