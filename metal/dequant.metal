// GPU Q8_0 → plain INT8 conversion kernel.
// Each thread handles one Q8_0 block (32 elements).  First pass writes
// dequantized FP32 to a temp buffer + per-block max to a reduction buffer.
// The host reads the reduction buffer, computes global scale, then calls
// the requantization kernel.

#include <metal_stdlib>
using namespace metal;

// block_q8_0 is defined in the base Metal source (half d + int8_t qs[32]).

// Pass 1: dequantize + write per-block max_abs
kernel void kernel_q8_to_i8_step1(
    device const block_q8_0 *src     [[buffer(0)]],
    device float            *absmax  [[buffer(1)]],  // one float per block
    uint   gid [[thread_position_in_grid]])
{
    half d = src[gid].d;
    float local_max = 0.0f;
    for (uint i = 0u; i < 32u; i++) {
        float v = (float)(int8_t)src[gid].qs[i] * (float)d;
        float av = v < 0.0f ? -v : v;
        if (av > local_max) local_max = av;
    }
    absmax[gid] = local_max;
}

// Pass 2: requantize using the global uniform scale.
// dst = round(src[gid].qs[i] * d[gid] / single_scale)
kernel void kernel_q8_to_i8_step2(
    device const block_q8_0 *src     [[buffer(0)]],
    device int8_t           *dst     [[buffer(1)]],
    constant float         &single_scale [[buffer(2)]],
    uint   gid [[thread_position_in_grid]])
{
    half d = src[gid].d;
    float inv = 1.0f / single_scale;
    uint base = gid * 32u;
    for (uint i = 0u; i < 32u; i++) {
        float v = (float)(int8_t)src[gid].qs[i] * (float)d;
        int val = (int)(v * inv + (v >= 0.0f ? 0.5f : -0.5f));
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        dst[base + i] = (int8_t)val;
    }
}
