// DS4 Neural Accelerator tensor matmul kernels for M5/A19+.
//
// kernel_tensor_mm_f16_f32 : half   weight × float activation → float output
// kernel_tensor_mm_i8_f32  : int8_t weight × float activation → float output
//
// X is (n_tok, in_dim) float activations.
// W is (in_dim, out_dim), row-major.  The host-wrapped MTLTensor uses
// dextents<in_dim, out_dim> stride {1, in_dim} for both half and int8_t.

#if defined(__METAL_VERSION__) && defined(__HAVE_TENSOR__)

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

struct args {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    float   post_scale;  // 1.0 for FP16, single_scale for INT8
};

// ---- FP16 weight -----------------------------------------------------------------
kernel void kernel_tensor_mm_f16_f32(
    constant args &a                                      [[buffer(0)]],
    tensor<device half,  dextents<int32_t, 2>> w [[buffer(1)]],
    tensor<device float, dextents<int32_t, 2>> x [[buffer(2)]],
    tensor<device float, dextents<int32_t, 2>> c [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            64, 32, static_cast<int>(metal::dynamic_extent),
            false, false, false),
        metal::execution_simdgroups<4>> op;

    int m_start = (int)tgid.y * 64;
    int n_start = (int)tgid.x * 32;
    auto xs = x.slice(0, m_start);
    auto ws = w.slice(0, n_start);
    auto cs = c.slice(n_start, m_start);
    op.run(xs, ws, cs);
}

// ---- INT8 weight -----------------------------------------------------------------
kernel void kernel_tensor_mm_i8_f32(
    constant args &a                                         [[buffer(0)]],
    tensor<device int8_t, dextents<int32_t, 2>> w [[buffer(1)]],
    tensor<device float,  dextents<int32_t, 2>> x [[buffer(2)]],
    tensor<device float,  dextents<int32_t, 2>> c [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            64, 32, static_cast<int>(metal::dynamic_extent),
            false, false, false),
        metal::execution_simdgroups<4>> op;

    int m_start = (int)tgid.y * 64;
    int n_start = (int)tgid.x * 32;
    auto xs = x.slice(0, m_start);
    auto ws = w.slice(0, n_start);
    auto cs = c.slice(n_start, m_start);
    op.run(xs, ws, cs);

    if (a.post_scale != 1.0f) {
        float s = a.post_scale;
        int max_m = a.ne1, max_n = a.ne0;
        for (int m = m_start; m < m_start + 64 && m < max_m; m++) {
            for (int n = n_start; n < n_start + 32 && n < max_n; n++) {
                c[array<int, 2>{n, m}] *= s;
            }
        }
    }
}

#endif
