#ifndef NPU_SRAM_H
#define NPU_SRAM_H

#include "npu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// SRAM Configuration: 4 Clusters, each with 512KB SRAM
#define NPU_SRAM_PER_CLUSTER (512 * 1024) // 512KB per cluster
#define NPU_SRAM_SIZE (NPU_SRAM_PER_CLUSTER * 4) // 2MB total (4 clusters)

// Initialize SRAM
void sram_init();

// Allocate memory in SRAM (simple bump allocator for model)
void* sram_malloc(size_t size);

// Reset SRAM (for new runs)
void sram_reset();

// Load/Store interface (Cycle counting)
// Dest, Src, Size, Time Accumulator
void sram_load(void* dest, const void* src, size_t size, int* t);
void sram_store(void* dest, const void* src, size_t size, int* t);

#ifdef __cplusplus
}
#endif

#endif // NPU_SRAM_H

