// Fused FP8 feed-forward block for the FastConformer encoder on GB10 (sm_121, built for the sm_120 family).
//
// Hypothesis: TensorRT runs SiLU + FP8 quantise between the two FF GEMMs as a separate memory-bound kernel (about
// 0.35 ms of 1.6 ms per FF block at 32 x 16 s, about 17 ms per batch over 48 blocks) and never fuses it into the
// GEMM epilogue. Doing it in the first GEMM's epilogue removes a 52 MB fp16 write and read per block.
//   GEMM1: H8 = fp8( silu(alpha1 * (X8 @ W1^T)) * oscale )      [M, N1] e4m3
//   GEMM2: Y  = fp16( alpha2 * (H8 @ W2^T) )                     [M, N2]
// X8 [M, K1] e4m3 row-major, W1 [N1, K1] and W2 [N2, N1] e4m3 (nn.Linear layout, K contiguous). No biases (the
// model's FF linears have none). Ablate by exporting without the plugin (lean_export.py, LEAN_FFN_PLUGIN=0).
//
// CUTLASS 4.1 (flashinfer's copy in the NeMo 25.11 container), CollectiveBuilder for Sm120 with an EVT epilogue.
// Exposes ffn_fp8_run() for a ctypes test and the TensorRT plugin "FFNFp8" (ffn_plugin.cpp).

#include <cuda_runtime.h>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

using E4M3 = cutlass::float_e4m3_t;
using Acc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int kAlign8 = 16;  // 128-bit accesses of 8-bit elements
constexpr int kAlign16 = 8;  // of 16-bit elements
#if defined(FC_SM100)
using Arch = cutlass::arch::Sm100;    // datacentre Blackwell (B200, GB200)
#elif defined(FC_SM90)
using Arch = cutlass::arch::Sm90;     // Hopper (GH200 / H100): same TMA warp-specialised FP8 kernels
#else
using Arch = cutlass::arch::Sm120;    // Blackwell sm_120 family (DGX Spark GB10, RTX Pro)
#endif
using OpClass = cutlass::arch::OpClassTensorOp;
using Cluster = Shape<_1, _1, _1>;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;

namespace fus = cutlass::epilogue::fusion;
// Per-architecture schedules. sm_120 accepts Auto epilogues with fused (EVT) epilogues; sm_90 needs the explicit TMA
// warp-specialised epilogue matching the mainloop; sm_100 (datacentre Blackwell) runs every GEMM as a 1-SM UMMA kernel.
#if defined(FC_SM100)
template <class MainSched> struct MainFor { using type = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100; };
template <class MainSched> struct EpiFor { using type = cutlass::epilogue::TmaWarpSpecialized1Sm; };
#else
template <class MainSched> struct MainFor { using type = MainSched; };
template <class MainSched> struct EpiFor { using type = cutlass::epilogue::collective::EpilogueScheduleAuto; };
#if defined(FC_SM90)
template <> struct EpiFor<cutlass::gemm::KernelTmaWarpSpecializedCooperative> {
  using type = cutlass::epilogue::TmaWarpSpecializedCooperative;
};
template <> struct EpiFor<cutlass::gemm::KernelTmaWarpSpecializedPingpong> {
  using type = cutlass::epilogue::TmaWarpSpecialized;
};
#endif
#endif


// GEMM1 epilogue: D = e4m3( silu(alpha * acc) * oscale )
using ScaledAcc = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiplies, Acc, Acc, kRound>,
                               fus::Sm90ScalarBroadcast<Acc>, fus::Sm90AccFetch>;
using SiluNode = fus::Sm90EVT<fus::Sm90Compute<cutlass::epilogue::thread::SiLu, Acc, Acc, kRound>, ScaledAcc>;
using Epi1 = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiplies, E4M3, Acc, kRound>, SiluNode,
                          fus::Sm90ScalarBroadcast<Acc>>;
// GEMM2 epilogue: D = fp16( alpha * acc )
using Epi2 = fus::LinearCombination<cutlass::half_t, Acc, void, Acc>;

// GEMM2 with the half-step residual add folded in: D = res_scale * alpha * acc + residual (C = residual, same type as
// D: TensorRT keeps the residual stream in fp32, so both fp32 and fp16 variants)
template <class E, class TileT = Shape<_128, _128, _64>, class SchedT = cutlass::gemm::KernelTmaWarpSpecializedCooperative,
          class ClusterT = Cluster>
struct GemmRes {
  using Tile = TileT;
  static constexpr int AlignE = 128 / cutlass::sizeof_bits<E>::value;
  using Fusion = fus::LinearCombination<E, Acc, E, Acc>;
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OpClass, Tile, ClusterT, cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc,
      E, LayoutD, AlignE, E, LayoutD, AlignE, typename EpiFor<SchedT>::type, Fusion>::CollectiveOp;
  using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OpClass, E4M3, LayoutA, kAlign8, E4M3, LayoutB, kAlign8, Acc, Tile, ClusterT,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      typename MainFor<SchedT>::type>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollMain, CollEpi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// Biased variants (models whose FF linears have biases, e.g. parakeet 1.1B): the bias is a per-column row broadcast.
template <class Tile>
using BiasRow = fus::Sm90RowBroadcast<0, Tile, float, float, Stride<_0, _1, _0>>;
template <class Tile>   // GEMM1: e4m3( silu(alpha * acc + b1) * oscale )
using Epi1B = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiplies, E4M3, Acc, kRound>,
    fus::Sm90EVT<fus::Sm90Compute<cutlass::epilogue::thread::SiLu, Acc, Acc, kRound>,
        fus::Sm90EVT<fus::Sm90Compute<cutlass::multiply_add, Acc, Acc, kRound>, fus::Sm90ScalarBroadcast<Acc>,
                     fus::Sm90AccFetch, BiasRow<Tile>>>,
    fus::Sm90ScalarBroadcast<Acc>>;
template <class Tile>   // GEMM2: fp16( alpha * acc + b2 )
using Epi2B = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiply_add, cutlass::half_t, Acc, kRound>,
                           fus::Sm90ScalarBroadcast<Acc>, fus::Sm90AccFetch, BiasRow<Tile>>;

// GEMM2 + residual + bias: D = (alpha * acc + bias) + residual  (alpha and bias pre-scaled by res_scale)
template <class E>
struct GemmResB {
  using Tile = Shape<_128, _128, _64>;
  static constexpr int AlignE = 128 / cutlass::sizeof_bits<E>::value;
  using Fusion = fus::Sm90EVT<fus::Sm90Compute<cutlass::plus, E, Acc, kRound>,
                              fus::Sm90EVT<fus::Sm90Compute<cutlass::multiply_add, Acc, Acc, kRound>,
                                           fus::Sm90ScalarBroadcast<Acc>, fus::Sm90AccFetch, BiasRow<Tile>>,
                              fus::Sm90SrcFetch<E>>;
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc,
      E, LayoutD, AlignE, E, LayoutD, AlignE, typename EpiFor<cutlass::gemm::KernelTmaWarpSpecializedCooperative>::type, Fusion>::CollectiveOp;
  using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OpClass, E4M3, LayoutA, kAlign8, E4M3, LayoutB, kAlign8, Acc, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      typename MainFor<cutlass::gemm::KernelTmaWarpSpecializedCooperative>::type>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollMain, CollEpi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// Conv-module pw1 + GLU: D = (alpha * acc) * sigmoid(C), with C = the gate half computed by a first GEMM (fp16),
// so only the [M, D] GLU output is written instead of the [M, 2D] pre-GLU tensor.
struct GemmGlu {
  using Tile = Shape<_128, _128, _64>;
  using H = cutlass::half_t;
  using Value = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiplies, Acc, Acc, kRound>, fus::Sm90ScalarBroadcast<Acc>,
                             fus::Sm90AccFetch>;
  using Gate = fus::Sm90EVT<fus::Sm90Compute<cutlass::epilogue::thread::Sigmoid, Acc, Acc, kRound>, fus::Sm90SrcFetch<H>>;
  using Fusion = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiplies, H, Acc, kRound>, Value, Gate>;
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc,
      H, LayoutD, 8, H, LayoutD, 8, typename EpiFor<cutlass::gemm::KernelTmaWarpSpecializedCooperative>::type, Fusion>::CollectiveOp;
  using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OpClass, E4M3, LayoutA, kAlign8, E4M3, LayoutB, kAlign8, Acc, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      typename MainFor<cutlass::gemm::KernelTmaWarpSpecializedCooperative>::type>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollMain, CollEpi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class Tile, class Sched, class ElementD, int AlignD, class Fusion, class ClusterT = Cluster>
struct GemmT {
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OpClass, Tile, ClusterT, cutlass::epilogue::collective::EpilogueTileAuto, Acc, Acc,
      void, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      typename EpiFor<Sched>::type, Fusion>::CollectiveOp;
  using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OpClass, E4M3, LayoutA, kAlign8, E4M3, LayoutB, kAlign8, Acc, Tile, ClusterT,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      typename MainFor<Sched>::type>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollMain, CollEpi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
// Variants (tile, schedule); FFN_V1 / FFN_V2 pick the production pair.
using Cl11 = Shape<_1, _1, _1>;
using Cl21 = Shape<_2, _1, _1>;
using Cl12 = Shape<_1, _2, _1>;
// Variants (tile, schedule, cluster). 0-7 are the sm_120 set (single-CTA clusters only on GB10); 8-13 add the Hopper
// shapes (bigger tiles, 2-CTA clusters for TMA multicast). Chosen at run time: FFN_V1 (GEMM1), FFN_V2 (plain GEMM2),
// FFN_VR (GEMM2 with the residual folded in, -1 = the fixed 128x128x64 cooperative kernel), SPL_VR (dense linears).
#define FFN_VARIANTS_BASE(X) \
  X(0, Shape<_128 COMMA _128 COMMA _128>, Coop, Cl11) \
  X(1, Shape<_128 COMMA _128 COMMA _128>, Ping, Cl11) \
  X(2, Shape<_64 COMMA _128 COMMA _128>, Ping, Cl11) \
  X(3, Shape<_128 COMMA _256 COMMA _64>, Coop, Cl11) \
  X(4, Shape<_128 COMMA _128 COMMA _64>, Coop, Cl11) \
  X(5, Shape<_128 COMMA _128 COMMA _64>, Ping, Cl11) \
  X(6, Shape<_64 COMMA _256 COMMA _128>, Ping, Cl11) \
  X(7, Shape<_128 COMMA _64 COMMA _128>, Ping, Cl11)
#if defined(FC_SM90)
#define FFN_VARIANTS(X) FFN_VARIANTS_BASE(X) \
  X(8, Shape<_128 COMMA _256 COMMA _128>, Coop, Cl21) \
  X(9, Shape<_128 COMMA _256 COMMA _64>, Coop, Cl21) \
  X(10, Shape<_128 COMMA _128 COMMA _128>, Coop, Cl21) \
  X(11, Shape<_128 COMMA _128 COMMA _128>, Ping, Cl21) \
  X(12, Shape<_256 COMMA _128 COMMA _64>, Coop, Cl12) \
  X(13, Shape<_128 COMMA _256 COMMA _128>, Coop, Cl12)
#else
#define FFN_VARIANTS(X) FFN_VARIANTS_BASE(X)
#endif
#define COMMA ,

template <class Gemm>
typename Gemm::Arguments make_args(const void* a, const void* b, void* d, int M, int N, int K) {
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SC = typename Gemm::GemmKernel::StrideC;
  using SD = typename Gemm::GemmKernel::StrideD;
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {M, N, K, 1};
  args.mainloop.ptr_A = static_cast<typename Gemm::ElementA const*>(a);
  args.mainloop.dA = cutlass::make_cute_packed_stride(SA{}, make_shape(M, K, 1));
  args.mainloop.ptr_B = static_cast<typename Gemm::ElementB const*>(b);
  args.mainloop.dB = cutlass::make_cute_packed_stride(SB{}, make_shape(N, K, 1));
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = cutlass::make_cute_packed_stride(SC{}, make_shape(M, N, 1));
  args.epilogue.ptr_D = static_cast<typename Gemm::GemmKernel::ElementD*>(d);
  args.epilogue.dD = cutlass::make_cute_packed_stride(SD{}, make_shape(M, N, 1));
  return args;
}

template <class Gemm>
int run(typename Gemm::Arguments& args, void* ws, size_t ws_bytes, cudaStream_t s) {
  Gemm g;
  if (g.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (Gemm::get_workspace_size(args) > ws_bytes) return 2;
  if (g.initialize(args, ws, s) != cutlass::Status::kSuccess) return 3;
  if (g.run(s) != cutlass::Status::kSuccess) return 4;
  return 0;
}

}  // namespace

template <class G1>
int run_g1(const void* x8, void* h8, const void* w1, int M, int K1, int N1, float alpha1, float oscale, void* ws,
           size_t wsz, cudaStream_t s) {
  using Gemm = typename G1::Gemm;
  auto a = make_args<Gemm>(x8, w1, h8, M, N1, K1);
  // Epi1 = EVT<mul_to_e4m3, EVT<silu, EVT<mul, alpha, acc>>, oscale>: children first, node op last
  a.epilogue.thread = {{{{{alpha1}}, {}, {}}, {}}, {{oscale}}, {}};
  return run<Gemm>(a, ws, wsz, s);
}

template <class G2>
int run_g2(const void* h8, void* y, const void* w2, int M, int N1, int N2, float alpha2, void* ws, size_t wsz,
           cudaStream_t s) {
  using Gemm = typename G2::Gemm;
  auto a = make_args<Gemm>(h8, w2, y, M, N2, N1);
  a.epilogue.thread.alpha = alpha2;
  a.epilogue.thread.beta = 0.f;
  return run<Gemm>(a, ws, wsz, s);
}

static int env_int(const char* name, int def) {   // run-time variant choice (sweeps without rebuilding)
  const char* v = getenv(name);
  return v && *v ? atoi(v) : def;
}

// y = residual + res_scale * alpha * a @ b^T with GEMM variant v (-1: GemmRes default tile)
template <class E>
int run_res_v(int v, const void* a8, const void* b8, const void* res, void* y, int M, int N, int K, float alpha_eff,
              cudaStream_t s) {
  auto go = [&](auto g) {
    using Gemm = typename decltype(g)::Gemm;
    auto a = make_args<Gemm>(a8, b8, y, M, N, K);
    a.epilogue.ptr_C = static_cast<E const*>(res);
    a.epilogue.thread.alpha = alpha_eff;
    a.epilogue.thread.beta = 1.f;
    return run<Gemm>(a, nullptr, 0, s);
  };
#if defined(FC_SM90)                         // variants only on Hopper; GB10's shared memory fits the default only
  switch (v) {
#define X(i, T, S, C) case i: return go(GemmRes<E, T, S, C>{});
    FFN_VARIANTS(X)
#undef X
    default: return go(GemmRes<E>{});
  }
#else
  (void)v;
  return go(GemmRes<E>{});
#endif
}

extern "C" {

int ffn_gemm1(int v, const void* x8, void* h8, const void* w1, int M, int K1, int N1, float alpha1, float oscale,
              void* ws, size_t wsz, cudaStream_t s) {
  switch (v) {
#define X(i, T, S, C) case i: return run_g1<GemmT<T, S, E4M3, kAlign8, Epi1, C>>(x8, h8, w1, M, K1, N1, alpha1, oscale, ws, wsz, s);
    FFN_VARIANTS(X)
#undef X
  }
  return 99;
}

int ffn_gemm2(int v, const void* h8, void* y, const void* w2, int M, int N1, int N2, float alpha2, void* ws,
              size_t wsz, cudaStream_t s) {
  switch (v) {
#define X(i, T, S, C) case i: return run_g2<GemmT<T, S, cutlass::half_t, kAlign16, Epi2, C>>(h8, y, w2, M, N1, N2, alpha2, ws, wsz, s);
    FFN_VARIANTS(X)
#undef X
  }
  return 99;
}

// Production pair from sweep_ffn.py (M = 6400, GB10 at 1.05 GHz): GEMM1 ping-pong 128x128x128 0.663 ms (cuBLAS
// plain GEMM 0.732), GEMM2 cooperative 128x128x64 0.603 ms (cuBLAS 0.709).
#ifndef FFN_V1
#define FFN_V1 1
#endif
#ifndef FFN_V2
#define FFN_V2 4
#endif

// Returns 0 on success, otherwise 10 * stage + step code (stage 1 = GEMM1, 2 = GEMM2). CUTLASS workspace is 0 for
// these (non split-K) kernels; ws may be null.
int ffn_fp8_run(const void* x8, void* h8, void* y, const void* w1, const void* w2, int M, int K1, int N1, int N2,
                float alpha1, float oscale, float alpha2, void* ws, size_t ws_bytes, cudaStream_t s) {
  static const int v1 = env_int("FFN_V1", FFN_V1), v2 = env_int("FFN_V2", FFN_V2);
  int r = ffn_gemm1(v1, x8, h8, w1, M, K1, N1, alpha1, oscale, ws, ws_bytes, s);
  if (r) return 10 + r;
  r = ffn_gemm2(v2, h8, y, w2, M, N1, N2, alpha2, ws, ws_bytes, s);
  if (r) return 20 + r;
  return 0;
}

size_t ffn_fp8_cutlass_ws(int M, int K1, int N1, int N2) { return 0; }

// As ffn_fp8_run, plus the residual: y = residual + res_scale * FF(x). is_f32: residual and y are fp32 (else fp16).
int ffn_fp8_run_res(const void* x8, void* h8, const void* res, void* y, const void* w1, const void* w2, int M, int K1,
                    int N1, int N2, float alpha1, float oscale, float alpha2, float res_scale, int is_f32,
                    cudaStream_t s) {
  static const int v1 = env_int("FFN_V1", FFN_V1), vr = env_int("FFN_VR", -1);
  int r = ffn_gemm1(v1, x8, h8, w1, M, K1, N1, alpha1, oscale, nullptr, 0, s);
  if (r) return 10 + r;
  r = is_f32 ? run_res_v<float>(vr, h8, w2, res, y, M, N2, N1, res_scale * alpha2, s)
             : run_res_v<cutlass::half_t>(vr, h8, w2, res, y, M, N2, N1, res_scale * alpha2, s);
  return r ? 20 + r : 0;
}

// Dense FP8 linear with the residual add folded into the epilogue (attention output projection, conv pw2):
// y = residual + res_scale * alpha * x8 @ W^T in the residual's type (is_f32). W e4m3 [N, K] row-major.
int fp8_linear_res_run(const void* x8, const void* w, const void* res, void* y, int M, int N, int K, float alpha,
                       float res_scale, int is_f32, cudaStream_t s) {
  static const int vr = env_int("SPL_VR", -1);
  return is_f32 ? run_res_v<float>(vr, x8, w, res, y, M, N, K, res_scale * alpha, s)
                : run_res_v<cutlass::half_t>(vr, x8, w, res, y, M, N, K, res_scale * alpha, s);
}

// As fp8_linear_res_run with a bias: y = residual + res_scale * (alpha * x8 @ W^T + b); bs = b * res_scale on the
// device (fp32 [N]).
int fp8_linear_resb_run(const void* x8, const void* w, const float* bs, const void* res, void* y, int M, int N, int K,
                        float alpha, float res_scale, int is_f32, cudaStream_t s) {
  auto go = [&](auto tag) {
    using E = decltype(tag);
    using Gemm = typename GemmResB<E>::Gemm;
    auto a = make_args<Gemm>(x8, w, y, M, N, K);
    a.epilogue.ptr_C = static_cast<E const*>(res);
    a.epilogue.thread = {{{res_scale * alpha}, {}, {bs, 0.f, {}}, {}}, {}, {}};
    return run<Gemm>(a, nullptr, 0, s);
  };
  return is_f32 ? go(float{}) : go(cutlass::half_t{});
}

// pw1 + GLU without biases: w e4m3 [2N, K] (rows 0..N-1 value, N..2N-1 gate), y fp16 [M, N] =
// (alpha * x8 @ Wv^T) * sigmoid(alpha * x8 @ Wg^T); gate fp16 [M, N] scratch.
int fp8_pw_glu_run(const void* x8, const void* w, void* gate, void* y, int M, int N, int K, float alpha,
                   cudaStream_t s) {
  using GG = typename GemmRes<cutlass::half_t>::Gemm;
  auto ag = make_args<GG>(x8, static_cast<const uint8_t*>(w) + static_cast<size_t>(N) * K, gate, M, N, K);
  ag.epilogue.thread.alpha = alpha;
  ag.epilogue.thread.beta = 0.f;
  int r = run<GG>(ag, nullptr, 0, s);
  if (r) return 10 + r;
  using G = GemmGlu::Gemm;
  auto a = make_args<G>(x8, w, y, M, N, K);
  a.epilogue.ptr_C = static_cast<cutlass::half_t const*>(gate);
  a.epilogue.thread = {{{alpha}, {}, {}}, {{}, {}}, {}};
  r = run<G>(a, nullptr, 0, s);
  return r ? 20 + r : 0;
}

// Biased FF block (b1 [N1], b2 [N2] fp32 device pointers). res == nullptr: y = FF(x) fp16; else
// y = residual + res_scale * FF(x) in the residual's type (is_f32).
int ffn_fp8_run_bias(const void* x8, void* h8, const void* res, void* y, const void* w1, const void* w2,
                     const float* b1, const float* b2s, int M, int K1, int N1, int N2, float alpha1, float oscale,
                     float alpha2s, int is_f32, cudaStream_t s) {
  {
    using Tile = Shape<_128, _128, _128>;
    using Gemm = typename GemmT<Tile, Ping, E4M3, kAlign8, Epi1B<Tile>>::Gemm;
    auto a = make_args<Gemm>(x8, w1, h8, M, N1, K1);
    a.epilogue.thread = {{{{alpha1}, {}, {b1, 0.f, {}}, {}}, {}}, {{oscale}}, {}};
    int r = run<Gemm>(a, nullptr, 0, s);
    if (r) return 10 + r;
  }
  if (!res) {
    using Tile = Shape<_128, _128, _64>;
    using Gemm = typename GemmT<Tile, Coop, cutlass::half_t, kAlign16, Epi2B<Tile>>::Gemm;
    auto a = make_args<Gemm>(h8, w2, y, M, N2, N1);
    a.epilogue.thread = {{alpha2s}, {}, {b2s, 0.f, {}}, {}};
    int r = run<Gemm>(a, nullptr, 0, s);
    return r ? 20 + r : 0;
  }
  auto go = [&](auto tag) {
    using E = decltype(tag);
    using Gemm = typename GemmResB<E>::Gemm;
    auto a = make_args<Gemm>(h8, w2, y, M, N2, N1);
    a.epilogue.ptr_C = static_cast<E const*>(res);
    a.epilogue.thread = {{{alpha2s}, {}, {b2s, 0.f, {}}, {}}, {}, {}};
    return run<Gemm>(a, nullptr, 0, s);
  };
  int r = is_f32 ? go(float{}) : go(cutlass::half_t{});
  return r ? 30 + r : 0;
}

}  // extern "C"
