#include "npu_sram.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Global SRAM buffer
static uint8_t* g_sram_buffer = NULL;
static size_t g_sram_offset = 0;

void sram_init() {
    if (!g_sram_buffer) {
        g_sram_buffer = (uint8_t*)malloc(NPU_SRAM_SIZE);
        if (!g_sram_buffer) {
            fprintf(stderr, "Error: Failed to allocate NPU SRAM\n");
            exit(1);
        }
    }
    g_sram_offset = 0;
}

void sram_reset() {
    g_sram_offset = 0;
}

void* sram_malloc(size_t size) {
    if (!g_sram_buffer) sram_init();
    
    // Simple 64-byte alignment
    size_t aligned_size = (size + 63) & ~63;
    
    if (g_sram_offset + aligned_size > NPU_SRAM_SIZE) {
        fprintf(stderr, "Error: SRAM OOM\n");
        return NULL;
    }
    
    void* ptr = g_sram_buffer + g_sram_offset;
    g_sram_offset += aligned_size;
    return ptr;
}

void sram_load(void* dest, const void* src, size_t size, int* t) {
    // In a functional C-model, we assume pointers are valid.
    
    if (dest && src) {
        memcpy(dest, src, size);
    }
    
    // Update stats
    g_npu_stats.inst_sram++;
    // Cycle = 0: sram_load can be pipelined/overlapped with matmul in real hardware
    (void)t;  // No cycle cost
}

void sram_store(void* dest, const void* src, size_t size, int* t) {
    if (dest && src) {
        memcpy(dest, src, size);
    }
    
    g_npu_stats.inst_sram++;
    // Cycle = 0: sram_store can be pipelined/overlapped with matmul in real hardware
    (void)t;  // No cycle cost
}

