#ifndef NPU_DMA_H
#define NPU_DMA_H

#include "npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// DMA Copy function
// Copies data between two tensors (usually Host <-> SRAM)
// output cycle count t
void dma_copy(npu_tensor_t* dst, const npu_tensor_t* src, int* t);

#ifdef __cplusplus
}
#endif

#endif // NPU_DMA_H

