// NVFP4 (e2m1 values, ue4m3 scale per 16 along K, fp32 global scale) feed-forward GEMMs for GB10 (sm_120 family),
// CUTLASS 4.1 block-scaled tensor ops. Track A (stock weights, post-training quantisation): FP4 tensor cores run at
// about 1.5x the FP8 rate here (torch _scaled_mm: 130 vs 86 TFLOP/s at a locked clock), so the FF blocks (~45% of
// the encoder) are the target. Accuracy cost measured separately (nvfp4_probe.py).
//   quant:  x fp16 [R, K] -> packed e2m1 [R, K/2] + scale factors in CUTLASS's swizzled layout (via its own layout
//           object, so the GEMMs and the quantiser cannot disagree); value = e2m1 * sf * g.
//   gemm1:  h = silu(alpha * x @ W1^T) written as NVFP4 (+ its scale factors) by the epilogue
//   gemm2:  y = alpha * h @ W2^T (+ residual) fp16
// Test entry points only for now (test_nvfp4.py); plugin wiring follows if the speed is there.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace nv4 {
using E2M1 = cutlass::float_e2m1_t;
using UE4M3 = cutlass::float_ue4m3_t;
using ElementAB = cutlass::nv_float4_t<E2M1>;
using Half = cutlass::half_t;
using Arch = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
constexpr int SFVec = 16;
namespace fus = cutlass::epilogue::fusion;

template <class ElementD, class ElementC, class Fusion>
struct G {
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignC = cute::is_void_v<ElementC> ? AlignD : 128 / cutlass::sizeof_bits<ElementC>::value;
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementC, cutlass::layout::RowMajor, AlignC, ElementD, cutlass::layout::RowMajor, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
  using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OpClass, ElementAB, cutlass::layout::RowMajor, 32, ElementAB, cutlass::layout::ColumnMajor, 32, float,
      Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollMain, CollEpi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Cfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};
// sm120 (non ptr-array) specialises the activation + block-scale epilogue only in its per-column-bias form; a null
// bias pointer reads as zero
using F1 = fus::LinCombPerColBiasEltActBlockScaleFactor<cutlass::epilogue::thread::SiLu, SFVec, E2M1, float, UE4M3,
                                                        cutlass::layout::RowMajor, float, void>;
using G1 = G<E2M1, void, F1>;
using G2 = G<Half, Half, fus::LinearCombination<Half, float, Half, float>>;
using Cfg = G1::Cfg;

// one thread per 16-element block: scale factor via CUTLASS's layout, two e2m1 per byte (low nibble first)
template <class LayoutSF>
__global__ void quant_kernel(const __half* __restrict__ x, uint8_t* __restrict__ q, UE4M3* __restrict__ sf, int R,
                             int K, float inv_g, LayoutSF layout_sf) {
  int blk = blockIdx.x * blockDim.x + threadIdx.x;
  int nb = K / SFVec;
  if (blk >= R * nb) return;
  int r = blk / nb, kb = blk % nb;
  const __half* src = x + static_cast<size_t>(r) * K + kb * SFVec;
  float v[SFVec], amax = 0.f;
#pragma unroll
  for (int i = 0; i < SFVec; ++i) { v[i] = __half2float(src[i]) * inv_g; amax = fmaxf(amax, fabsf(v[i])); }
  UE4M3 s = UE4M3(amax / 6.f);
  float sc = float(s);
  float inv = sc > 0.f ? 1.f / sc : 0.f;
  uint8_t* dst = q + (static_cast<size_t>(r) * K + kb * SFVec) / 2;
#pragma unroll
  for (int i = 0; i < SFVec; i += 2) {
    uint8_t lo = E2M1(v[i] * inv).raw() & 0xF, hi = E2M1(v[i + 1] * inv).raw() & 0xF;
    dst[i / 2] = lo | (hi << 4);
  }
  sf[layout_sf(r, kb * SFVec, 0)] = s;              // layout over (rows, K, L): stride 0 within a 16-block
}
}  // namespace nv4

extern "C" {

size_t nvfp4_sf_bytes(int R, int K) {           // generous: rows padded to 128, K blocks to 4
  return static_cast<size_t>((R + 127) / 128 * 128) * ((K / 16 + 3) / 4 * 4);
}

// activations (A operand, rows = M) and weights (B operand, rows = N) share the K-major atom
int nvfp4_quant(const void* x, void* q, void* sf, int R, int K, float g, int as_b, cudaStream_t s) {
  using namespace nv4;
  cudaMemsetAsync(sf, 0, nvfp4_sf_bytes(R, K), s);
  int threads = 256, blocks = (R * (K / SFVec) + threads - 1) / threads;
  if (as_b) {
    auto l = Cfg::tile_atom_to_shape_SFB(make_shape(1, R, K, 1));
    quant_kernel<<<blocks, threads, 0, s>>>(static_cast<const __half*>(x), static_cast<uint8_t*>(q),
                                            static_cast<UE4M3*>(sf), R, K, 1.f / g, l);
  } else {
    auto l = Cfg::tile_atom_to_shape_SFA(make_shape(R, 1, K, 1));
    quant_kernel<<<blocks, threads, 0, s>>>(static_cast<const __half*>(x), static_cast<uint8_t*>(q),
                                            static_cast<UE4M3*>(sf), R, K, 1.f / g, l);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

// h = NVFP4(silu(alpha * x @ W1^T)) with its scale factors (norm_constant: device float, the output global scale)
int nvfp4_gemm1(const void* xq, const void* sfx, const void* wq, const void* sfw, void* hq, void* sfh, int M, int N,
                int K, float alpha, const float* norm_constant, cudaStream_t s) {
  using namespace nv4;
  using Gm = G1::Gemm;
  using Kn = G1::Kernel;
  typename Gm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {M, N, K, 1};
  args.mainloop.ptr_A = static_cast<typename Kn::ElementA const*>(xq);
  args.mainloop.dA = cutlass::make_cute_packed_stride(typename Kn::StrideA{}, make_shape(M, K, 1));
  args.mainloop.ptr_B = static_cast<typename Kn::ElementB const*>(wq);
  args.mainloop.dB = cutlass::make_cute_packed_stride(typename Kn::StrideB{}, make_shape(N, K, 1));
  args.mainloop.ptr_SFA = static_cast<UE4M3 const*>(sfx);
  args.mainloop.layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  args.mainloop.ptr_SFB = static_cast<UE4M3 const*>(sfw);
  args.mainloop.layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  args.epilogue.ptr_D = static_cast<E2M1*>(hq);
  args.epilogue.dD = cutlass::make_cute_packed_stride(typename Kn::StrideD{}, make_shape(M, N, 1));
  args.epilogue.thread.alpha = alpha;
  args.epilogue.thread.block_scale_factor_ptr = static_cast<UE4M3*>(sfh);
  args.epilogue.thread.norm_constant_ptr = norm_constant;
  args.epilogue.thread.bias_ptr = nullptr;
  Gm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (Gm::get_workspace_size(args) > 0) return 2;
  if (op.initialize(args, nullptr, s) != cutlass::Status::kSuccess) return 3;
  return op.run(s) == cutlass::Status::kSuccess ? 0 : 4;
}

// y fp16 = alpha * h @ W2^T (+ beta * residual)
int nvfp4_gemm2(const void* hq, const void* sfh, const void* wq, const void* sfw, const void* res, void* y, int M,
                int N, int K, float alpha, float beta, cudaStream_t s) {
  using namespace nv4;
  using Gm = G2::Gemm;
  using Kn = G2::Kernel;
  typename Gm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {M, N, K, 1};
  args.mainloop.ptr_A = static_cast<typename Kn::ElementA const*>(hq);
  args.mainloop.dA = cutlass::make_cute_packed_stride(typename Kn::StrideA{}, make_shape(M, K, 1));
  args.mainloop.ptr_B = static_cast<typename Kn::ElementB const*>(wq);
  args.mainloop.dB = cutlass::make_cute_packed_stride(typename Kn::StrideB{}, make_shape(N, K, 1));
  args.mainloop.ptr_SFA = static_cast<UE4M3 const*>(sfh);
  args.mainloop.layout_SFA = G2::Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  args.mainloop.ptr_SFB = static_cast<UE4M3 const*>(sfw);
  args.mainloop.layout_SFB = G2::Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  args.epilogue.ptr_C = static_cast<Half const*>(res);
  args.epilogue.dC = cutlass::make_cute_packed_stride(typename Kn::StrideC{}, make_shape(M, N, 1));
  args.epilogue.ptr_D = static_cast<Half*>(y);
  args.epilogue.dD = cutlass::make_cute_packed_stride(typename Kn::StrideD{}, make_shape(M, N, 1));
  args.epilogue.thread.alpha = alpha;
  args.epilogue.thread.beta = res ? beta : 0.f;
  Gm op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (Gm::get_workspace_size(args) > 0) return 2;
  if (op.initialize(args, nullptr, s) != cutlass::Status::kSuccess) return 3;
  return op.run(s) == cutlass::Status::kSuccess ? 0 : 4;
}

}  // extern "C"
