// DS4 ANE-accelerated FP16 tensor matmul for attention projections.
// Requires MTL4 tensor API (macOS 26.0+, M5+).  Both weight and activation
// are half; output is float.  Layout matches kernel_mul_mm_mpp_direct_rhs in
// dense.metal (C[out_dim, n_tok] = W[out_dim, in_dim] * X[n_tok, in_dim]^T).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct tensor_mm_args {
    int32_t ne00;
    int32_t ne0;
    int32_t ne1;
    uint32_t nb01;
    float   post_scale;
};

kernel void kernel_tensor_mm_f16_f32(
    constant tensor_mm_args &args [[buffer(0)]],
    device const char *src0 [[buffer(1)]],
    device const char *src1 [[buffer(2)]],
    device       char *dst  [[buffer(3)]],
    threadgroup  char *shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NR1 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const bool full_tile = r0 + NR0 <= M && r1 + NR1 <= N && (K % NK) == 0;

    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device half *ptrB = (device half *)((device char *)src1);
    const int strideB = (int)args.nb01;
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (full_tile || r0 + row < M) {
                device const half *row_ptr =
                    (device const half *)(src0 + (uint64_t)(r0 + row) * args.nb01);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] =
                        (full_tile || k_pos + i < K) ? row_ptr[k_pos + i] : (half)0;
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_batch = (device float *)dst;
    if (full_tile) {
        device float *dst_tile = dst_batch + r0 + (uint64_t)r1 * M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_batch, dextents<int32_t, 2>(M, N), array<int, 2>({1, M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }

    if (args.post_scale != 1.0f) {
        const float s = args.post_scale;
        for (int t = r1; t < r1 + NR1 && t < N; t++) {
            for (int o = r0; o < r0 + NR0 && o < M; o++) {
                dst_batch[(uint64_t)o + (uint64_t)t * M] *= s;
            }
        }
    }
}
