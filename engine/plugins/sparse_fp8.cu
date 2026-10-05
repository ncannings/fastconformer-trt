// 2:4 structured-sparse FP8 GEMM on GB10 (sm_120 family) with CUTLASS 4.1, as a feasibility test for pruning the
// encoder's linears. y[M, N] = x[M, K] @ W[N, K]^T with W 2:4-sparse along K: computed as D[N, M] = W @ x^T with the
// sparse operand A = W (row-major, K contiguous), B = x (column-major K x M), D column-major (= y row-major), fp16 out.
// torch's 2:4 paths are unusable here (cuSPARSELt runs 4-14x slower than dense on sm_121; its CUTLASS path is sm_8x).
// Exposes: sparse_fp8_prepare (compress W, returns handle) and sparse_fp8_run.

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace sp {
using E4M3 = cutlass::float_e4m3_t;
using Half = cutlass::half_t;
using Arch = cutlass::arch::Sm120;
using Tile = Shape<_128, _128, _256>;
using Cluster = Shape<_1, _1, _1>;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutDTag = cutlass::layout::ColumnMajor;

using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassSparseTensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, void, LayoutDTag, 8, Half, LayoutDTag, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassSparseTensorOp, E4M3, LayoutATag, 32, E4M3, LayoutBTag, 16, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using ProblemShape = Shape<int, int, int, int>;
using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollMain, CollEpi, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

using LayoutA = typename CollMain::LayoutA;
using LayoutE = typename CollMain::LayoutE;
using SparseConfig = typename CollMain::SparseConfig;
using StrideA = cutlass::gemm::TagToStrideA_t<LayoutATag>;
using StrideB = typename Kernel::StrideB;
using StrideD = typename Kernel::StrideD;
using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<ProblemShape, E4M3, LayoutATag,
                                                                                       SparseConfig>;
using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<ProblemShape, E4M3, LayoutATag,
                                                                                SparseConfig, Arch>;
using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

namespace fus = cutlass::epilogue::fusion;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;
// FF GEMM1 epilogue (transposed problem, D = h^T column-major = h row-major): e4m3( silu(alpha * acc) * oscale )
using EpiSilu = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiplies, E4M3, float, kRound>,
    fus::Sm90EVT<fus::Sm90Compute<cutlass::epilogue::thread::SiLu, float, float, kRound>,
        fus::Sm90EVT<fus::Sm90Compute<cutlass::multiplies, float, float, kRound>, fus::Sm90ScalarBroadcast<float>,
                     fus::Sm90AccFetch>>,
    fus::Sm90ScalarBroadcast<float>>;

template <class ElementC, class ElementD, int AlignCD, class Fusion, class TileT = Tile>
struct SpGemm {
  using Tile = TileT;
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassSparseTensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, ElementC, LayoutDTag, AlignCD, ElementD, LayoutDTag, AlignCD,
      cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
  using CollMainS = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassSparseTensorOp, E4M3, LayoutATag, 32, E4M3, LayoutBTag, 16, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using KernelS = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollMainS, CollEpi, void>;
  using G = cutlass::gemm::device::GemmUniversalAdapter<KernelS>;
};
using Gemm1S = SpGemm<void, E4M3, 16, EpiSilu>::G;
template <class E>
using Gemm2R = typename SpGemm<E, E, 128 / cutlass::sizeof_bits<E>::value, fus::LinearCombination<E, float, E, float>,
                               Shape<_128, _128, _128>>::G;   // C loads need smem: shallower K tile

using EpiQ = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiply_add, Half, float, kRound>, fus::Sm90ScalarBroadcast<float>,
                          fus::Sm90AccFetch,
                          fus::Sm90ColBroadcast<0, Tile, float, float, Stride<_1, _0, int64_t>>>;
using GemmQ = SpGemm<void, Half, 8, EpiQ>::G;

struct Handle {
  int N, K, L = 1;
  void* a_comp = nullptr;
  void* e = nullptr;
  LayoutA layout_a;
  LayoutE layout_e;
};
template <class G>
typename G::Arguments sp_args(Handle* h, const void* x, void* y, int M) {
  using K_ = typename G::GemmKernel;
  typename G::Arguments a{};
  a.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  a.problem_shape = ProblemShape{h->N, M, h->K, 1};
  a.mainloop.ptr_A = static_cast<E4M3 const*>(h->a_comp);
  a.mainloop.layout_a = h->layout_a;
  a.mainloop.ptr_B = static_cast<E4M3 const*>(x);
  a.mainloop.dB = cutlass::make_cute_packed_stride(typename K_::StrideB{}, cute::make_shape(M, h->K, 1));
  a.mainloop.ptr_E = static_cast<typename K_::CollectiveMainloop::ElementE const*>(h->e);
  a.mainloop.layout_e = h->layout_e;
  a.epilogue.ptr_D = static_cast<typename K_::ElementD*>(y);
  a.epilogue.dD = cutlass::make_cute_packed_stride(typename K_::StrideD{}, cute::make_shape(h->N, M, 1));
  a.epilogue.dC = cutlass::make_cute_packed_stride(typename K_::StrideC{}, cute::make_shape(h->N, M, 1));
  return a;
}

template <class G>
int sp_run(typename G::Arguments& a, cudaStream_t s) {
  G g;
  if (g.can_implement(a) != cutlass::Status::kSuccess) return 1;
  if (G::get_workspace_size(a) > 0) return 2;
  if (g.initialize(a, nullptr, s) != cutlass::Status::kSuccess) return 3;
  return g.run(s) == cutlass::Status::kSuccess ? 0 : 4;
}
}  // namespace sp

extern "C" {

// Sparse FF block: h8 = e4m3(silu(alpha1 * x8 @ W1^T) * oscale); y = residual + res_scale * alpha2 * (h8 @ W2^T)
// (residual fp32 if is_f32 else fp16; res == nullptr: y = alpha2 * h8 @ W2^T fp16). h1, h2 from sparse_fp8_prepare.
int sparse_ffn_run(void* h1, void* h2, const void* x8, void* h8, const void* res, void* y, int M, float alpha1,
                   float oscale, float alpha2, float res_scale, int is_f32, cudaStream_t s) {
  using namespace sp;
  auto a1 = sp_args<Gemm1S>(static_cast<Handle*>(h1), x8, h8, M);
  a1.epilogue.thread = {{{{{alpha1}}, {}, {}}, {}}, {{oscale}}, {}};
  int r = sp_run<Gemm1S>(a1, s);
  if (r) return 10 + r;
  auto go = [&](auto tag) {
    using E = decltype(tag);
    auto a2 = sp_args<Gemm2R<E>>(static_cast<Handle*>(h2), h8, y, M);
    a2.epilogue.ptr_C = static_cast<E const*>(res);
    a2.epilogue.thread.alpha = res ? res_scale * alpha2 : alpha2;
    a2.epilogue.thread.beta = res ? 1.f : 0.f;
    return sp_run<Gemm2R<E>>(a2, s);
  };
  r = (res && is_f32) ? go(float{}) : go(cutlass::half_t{});
  return r ? 20 + r : 0;
}

void* sparse_fp8_prepare_l(const void* W, int N, int K, int L, cudaStream_t s);

// W: dense e4m3 [N, K] (already 2:4 along K). Returns a handle with the compressed operand and metadata, or null.
void* sparse_fp8_prepare(const void* W, int N, int K, cudaStream_t s) { return sparse_fp8_prepare_l(W, N, K, 1, s); }

// Batched: W is L consecutive [N, K] blocks (e.g. per attention head, N = dk).
void* sparse_fp8_prepare_l(const void* W, int N, int K, int L, cudaStream_t s) {
  using namespace sp;
  auto h = new Handle();
  h->N = N;
  h->K = K;
  h->L = L;
  ProblemShape ps{N, 1, K, L};
  StrideA dA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(N, K, L));
  CompressorUtility util(ps, dA);
  size_t a_bytes = util.get_compressed_tensor_A_bytes();
  size_t e_bytes = util.get_tensor_E_bytes();
  if (cudaMalloc(&h->a_comp, a_bytes) != cudaSuccess || cudaMalloc(&h->e, e_bytes) != cudaSuccess) return nullptr;
  cudaMemset(h->a_comp, 0, a_bytes);
  cudaMemset(h->e, 0, e_bytes);
  h->layout_a = util.fill_layoutA_from_compressor();
  h->layout_e = util.fill_layoutE_from_compressor();
  cutlass::KernelHardwareInfo hw;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  typename Compressor::Arguments args{ps, {W, dA, h->a_comp, h->e}, hw};
  Compressor c;
  size_t ws = Compressor::get_workspace_size(args);
  void* wsp = nullptr;
  if (ws) cudaMalloc(&wsp, ws);
  if (c.can_implement(args) != cutlass::Status::kSuccess || c.initialize(args, wsp, s) != cutlass::Status::kSuccess ||
      c.run(s) != cutlass::Status::kSuccess)
    return nullptr;
  cudaStreamSynchronize(s);
  if (wsp) cudaFree(wsp);
  return h;
}

// y[M, N] fp16 = alpha * x[M, K] @ W^T. Returns 0 on success.
int sparse_fp8_run(void* handle, const void* x, void* y, int M, float alpha, cudaStream_t s) {
  using namespace sp;
  auto h = static_cast<Handle*>(handle);
  ProblemShape ps{h->N, M, h->K, 1};
  typename Gemm::Arguments a{};
  a.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  a.problem_shape = ps;
  a.mainloop.ptr_A = static_cast<E4M3 const*>(h->a_comp);
  a.mainloop.layout_a = h->layout_a;
  a.mainloop.ptr_B = static_cast<E4M3 const*>(x);
  a.mainloop.dB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(M, h->K, 1));
  a.mainloop.ptr_E = static_cast<typename CollMain::ElementE const*>(h->e);
  a.mainloop.layout_e = h->layout_e;
  a.epilogue.thread.alpha = alpha;
  a.epilogue.thread.beta = 0.f;
  a.epilogue.ptr_D = static_cast<Half*>(y);
  a.epilogue.dD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(h->N, M, 1));
  Gemm g;
  if (g.can_implement(a) != cutlass::Status::kSuccess) return 1;
  if (Gemm::get_workspace_size(a) > 0) return 2;
  if (g.initialize(a, nullptr, s) != cutlass::Status::kSuccess) return 3;
  if (g.run(s) != cutlass::Status::kSuccess) return 4;
  return 0;
}

// y[M, N] = alpha * x8 @ W^T (+ residual, y = residual + res_scale * alpha * x8 @ W^T, residual fp32 if is_f32).
int sparse_linear_run(void* h, const void* x8, const void* res, void* y, int M, float alpha, float res_scale,
                      int is_f32, cudaStream_t s) {
  using namespace sp;
  auto go = [&](auto tag) {
    using E = decltype(tag);
    auto a = sp_args<Gemm2R<E>>(static_cast<Handle*>(h), x8, y, M);
    a.epilogue.ptr_C = static_cast<E const*>(res);
    a.epilogue.thread.alpha = res ? res_scale * alpha : alpha;
    a.epilogue.thread.beta = res ? 1.f : 0.f;
    return sp_run<Gemm2R<E>>(a, s);
  };
  return (res && is_f32) ? go(float{}) : go(cutlass::half_t{});
}

// Heads-first sparse QKV: for part p in q, k, v: out_p [H, M, dk] fp16 = alpha3[p] * x8 @ W_p^T + bias_p, with each
// W_p compressed per head (sparse_fp8_prepare_l(W_p, dk, K, H)); bias fp32 [3, H * dk] (q part incl. pos_bias_u).
int sparse_qkv_run(void* h3[3], const void* x8, const float* bias, void* outs[3], int M, int H, int dk,
                   const float* alpha3, cudaStream_t s) {
  using namespace sp;
  for (int p = 0; p < 3; ++p) {
    auto hd = static_cast<Handle*>(h3[p]);
    using G = GemmQ;
    using K_ = typename G::GemmKernel;
    typename G::Arguments a{};
    a.mode = cutlass::gemm::GemmUniversalMode::kBatched;
    a.problem_shape = ProblemShape{dk, M, hd->K, H};
    a.mainloop.ptr_A = static_cast<E4M3 const*>(hd->a_comp);
    a.mainloop.layout_a = hd->layout_a;
    a.mainloop.ptr_B = static_cast<E4M3 const*>(x8);
    a.mainloop.dB = cutlass::make_cute_packed_stride(typename K_::StrideB{}, cute::make_shape(M, hd->K, H));
    get<2>(a.mainloop.dB) = 0;                                    // every head reads the same x
    a.mainloop.ptr_E = static_cast<typename K_::CollectiveMainloop::ElementE const*>(hd->e);
    a.mainloop.layout_e = hd->layout_e;
    a.epilogue.ptr_D = static_cast<Half*>(outs[p]);
    a.epilogue.dD = cutlass::make_cute_packed_stride(typename K_::StrideD{}, cute::make_shape(dk, M, H));
    a.epilogue.thread = {{alpha3[p]}, {}, {bias + static_cast<size_t>(p) * H * dk, 0.f, {_1{}, _0{}, int64_t(dk)}}, {}};
    int r = sp_run<G>(a, s);
    if (r) return 10 * (p + 1) + r;
  }
  return 0;
}

}  // extern "C"
