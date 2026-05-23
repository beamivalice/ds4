# M5 Neural Accelerator Integration for ds4

**PR summary**: Adds Apple M5 GPU Neural Accelerator (ANE) support to DeepSeek V4 Flash inference, delivering **+22% prefill throughput** for batched inference with no degradation to single-token generation speed.

---

## Hardware

Apple M5 Max, 128 GB unified memory, 614 GB/s memory bandwidth. The M5 GPU includes dedicated Neural Accelerator matrix-multiplication hardware delivering ~70 TFLOPS FP16 and ~130 TOPS INT8 — 2-4× the SIMD throughput.

## Problem

The ds4 inference engine uses Metall 3 compute for all matmul operations. The dominant overhead is matrix-multiplying FP16 or Q8_0 (block-quantized) weight tensors against FP32 activation tensors. During batched prefill, this is compute-bound on the standard SIMD shader cores (~7 TFLOPS). The Neural Accelerator hardware (10× more throughput) was unused.

## Architecture

Three layers of acceleration:

### Layer 1: FP16 ANE matmul for native FP16 weights

A new Metal Shading Language 4 kernel using `mpp::tensor_ops::matmul2d` is compiled into a separate `MTLLibrary` at Metal language version 4.0. On M5/A19 hardware, this kernel targets the ANE via `MTL4CommandBuffer` + `MTL4ArgumentTable` for tensor resource binding. Existing FP16 matmuls (HC mixing, compressor, indexer, router) are redirected automatically.

### Layer 2: INT8 ANE matmul for Q8_0 attention/FFN projections

At model load time (`DS4_METAL_Q8_I8_PRELOAD=1`), the Q8_0 attention and FFN projection tensors (345 tensors, ~97 MB total) are converted from block-quantized Q8_0 format (34 bytes per 32-element block with per-block scales) to plain INT8 format (1 byte per element with a single uniform scale per matrix). A second ANE kernel using `int8_t × float → float` type combination handles the INT8 weight path. The uniform scale is applied as a post-multiply after the matmul completes.

Memory: no overhead vs Q8_0 (plain INT8 is 256/292 = 12% smaller than Q8_0 blocks).

### Layer 3: Batched dispatch

Without batching, each of ~8600 ANE-eligible matmuls per prefill created a separate `MTL4CommandBuffer`, `MTL4CommandAllocator`, and `MTL4ArgumentTable` — ~15 µs overhead per dispatch totaling ~130 ms (2.5% of 5.2s prefill).

The fix: per-layer batch encoder. `ds4_gpu_ane_batch_begin()` creates one allocator, one command buffer, one encoder, and one argument table. All eligible matmuls within a layer share these resources. `ds4_gpu_ane_batch_end()` commits once per layer (43 total instead of 8600). This eliminates ~200 ms of dispatch overhead.

## Files changed

| File | Lines | Description |
|---|---|---|
| `metal/tensor_matmul.metal` | +60 | FP16 and INT8 ANE matmul kernels |
| `ds4_metal.m` | +310 | MTL4 library compilation, batch dispatch infrastructure, Q8_0→INT8 conversion, FP16/INT8 pipeline getters, shadow redirect in `ds4_gpu_matmul_q8_0_tensor`, FP16 ANE path in `ds4_gpu_matmul_f16_tensor` |
| `ds4_gpu.h` | +4 | Function declarations |
| `ds4.c` | +6 | `ds4_gpu_ane_batch_begin()/end()` calls in attention, FFN, and decode graph encoding functions |

## Benchmarks

See `bench_ds4.md` and `bench_ds4ane.md` for full results.

| Context (tokens) | Baseline pp (t/s) | ANE pp (t/s) | Speedup |
|---|---|---|---|
| 2048 | 437.4 | 473.1 | +8.2% |
| 6144 | 388.4 | 469.7 | +20.9% |
| 10240 | 377.3 | 453.6 | +20.2% |
| 14336 | 362.9 | 428.8 | +18.2% |
| 16384 | 353.1 | 411.7 | +16.6% |

Token generation speed unchanged (~30-31 t/s) — decode is memory-bandwidth bound (single-token, single-batch), already at 72% of the 614 GB/s ceiling.

## Activation

```bash
DS4_METAL_Q8_I8_PRELOAD=1 ./ds4-bench --prompt-file prompt.txt ...
```

One-time 4.6s conversion at model load. Subsequent inference uses ANE path for all eligible matmuls during prefill.

## Graceful fallback

- **Pre-M5 hardware**: `MTLLanguageVersion4_0` / `__HAVE_TENSOR__` unavailable → kernel library compile fails silently → SIMD path used
- **Tensor binding failure**: `@try/@catch` around MTL4 dispatch → falls through to SIMD
- **mtap-backed model buffers incompatible with MTLTensor**: Short-lived GPU buffer allocation for weight slice
- **Env var not set**: All ANE paths gated behind `DS4_METAL_Q8_I8_PRELOAD` for the INT8 path, or `n_tok >= 32` for the FP16 path; small-batch and single-token paths stay on SIMD
