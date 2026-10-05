// QKV projection that writes heads-first: the FP8 GEMM epilogue stores q_u, k and v directly as [H, B*T, dk] fp16
// with the projection bias (and pos_bias_u for q) added, so the attention needs no layout copy.
//
// Hypothesis: with heads-first attention, TensorRT spends about 10.5 ms per 32 x 16 s batch permuting q, k, v (and
// adding the positional biases) at memory bandwidth, whether q/k/v are one GEMM or three. Writing the heads-first
// layout from the GEMM epilogue removes that copy. q_v is not produced: bd = q_v.p = q_u.p + (b_v - b_u).p, and the
// second term is one [H, 2T-1] vector per layer (computed in the graph).
// Each of q, k, v is a batched GEMM over the H head blocks: A = X8 shared (batch stride 0), B = the head's dk rows
// of W, D = the head's [M, dk] block, bias broadcast per head. dk = 128 or 64 (one N tile per head; sparse24 needs 128).
// Ablate by exporting without LEAN_QKV_PLUGIN.

#include <cuda_runtime.h>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace qkvh {

using E4M3 = cutlass::float_e4m3_t;
using Half = cutlass::half_t;
namespace fus = cutlass::epilogue::fusion;
// Epilogue schedule per mainloop schedule: sm_120 accepts Auto with fused (EVT) epilogues; sm_90 needs the explicit
// TMA warp-specialised epilogue that matches the mainloop.
template <class MainSched> struct EpiFor { using type = cutlass::epilogue::collective::EpilogueScheduleAuto; };
#if defined(FC_SM90)
template <> struct EpiFor<cutlass::gemm::KernelTmaWarpSpecializedCooperative> {
  using type = cutlass::epilogue::TmaWarpSpecializedCooperative;
};
template <> struct EpiFor<cutlass::gemm::KernelTmaWarpSpecializedPingpong> {
  using type = cutlass::epilogue::TmaWarpSpecialized;
};
#endif

constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;
using Cluster = Shape<_1, _1, _1>;
#if defined(FC_SM90)
using ArchQ = cutlass::arch::Sm90;
#else
using ArchQ = cutlass::arch::Sm120;
#endif
using BiasStride = Stride<_0, _1, int64_t>;     // per column, one vector per batch (head)

// One N tile per head: TileN = dk (128 for the 0.6B/1.1B models, 64 for d_model 512 / 8 heads).
template <int TileN, int TileM = 128, int TileK = 128, class Sched = cutlass::gemm::KernelTmaWarpSpecializedPingpong>
struct G {
  using Tile = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  // D = alpha * acc + bias[n, l]
  using Epi = fus::Sm90EVT<fus::Sm90Compute<cutlass::multiply_add, Half, float, kRound>,
                           fus::Sm90ScalarBroadcast<float>, fus::Sm90AccFetch,
                           fus::Sm90RowBroadcast<0, Tile, float, float, BiasStride>>;
  using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchQ, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, cutlass::layout::RowMajor, 8, Half,
      cutlass::layout::RowMajor, 8, typename EpiFor<Sched>::type, Epi>::CollectiveOp;
  using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchQ, cutlass::arch::OpClassTensorOp, E4M3, cutlass::layout::RowMajor, 16, E4M3,
      cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollEpi::SharedStorage))>,
      Sched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollMain, CollEpi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class GG>
int run(const void* x8, const void* w8, const float* bias, void* const* outs, int M, int K, int H, int dk,
        const float* alpha3, cudaStream_t s) {
  using Gemm = typename GG::Gemm;
  using Kernel = typename GG::Kernel;
  for (int part = 0; part < 3; ++part) {
    typename Gemm::Arguments a{};
    a.mode = cutlass::gemm::GemmUniversalMode::kBatched;
    a.problem_shape = {M, dk, K, H};
    a.mainloop.ptr_A = static_cast<E4M3 const*>(x8);
    a.mainloop.dA = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(M, K, H));
    get<2>(a.mainloop.dA) = 0;                                       // every head reads the same X
    a.mainloop.ptr_B = static_cast<E4M3 const*>(w8) + static_cast<size_t>(part) * H * dk * K;
    a.mainloop.dB = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(dk, K, H));
    a.epilogue.ptr_C = nullptr;
    a.epilogue.dC = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, make_shape(M, dk, H));
    a.epilogue.ptr_D = static_cast<Half*>(outs[part]);
    a.epilogue.dD = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(M, dk, H));
    a.epilogue.thread = {{alpha3[part]}, {}, {bias + static_cast<size_t>(part) * H * dk, 0.f, {_0{}, _1{}, int64_t(dk)}}, {}};
    Gemm g;
    if (g.can_implement(a) != cutlass::Status::kSuccess) return 10 * (part + 1) + 1;
    if (g.initialize(a, nullptr, s) != cutlass::Status::kSuccess) return 10 * (part + 1) + 3;
    if (g.run(s) != cutlass::Status::kSuccess) return 10 * (part + 1) + 4;
  }
  return 0;
}

}  // namespace qkvh

extern "C" {

// x8 [M, K] e4m3; w8 [3*H*dk, K] e4m3 (rows q heads, k heads, v heads); bias [3*H*dk] fp32 (q part includes
// pos_bias_u); out_q/out_k/out_v [H, M, dk] fp16; alpha3 = s_x * s_w per part (q, k, v each have their own FP8
// weight scale). Returns 0, or 10 * part + step code.
int qkv_heads_run(const void* x8, const void* w8, const float* bias, void* out_q, void* out_k, void* out_v, int M,
                  int K, int H, int dk, const float* alpha3, cudaStream_t s) {
  using namespace qkvh;
  using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  void* outs[3] = {out_q, out_k, out_v};
  static const int v = getenv("QKV_V") ? atoi(getenv("QKV_V")) : 0;   // tile sweep (sweep with test_qkv.py)
  if (dk == 64) return run<G<64>>(x8, w8, bias, outs, M, K, H, dk, alpha3, s);
  if (dk != 128) return 1;
  switch (v) {
    case 1: return run<G<128, 128, 128, Coop>>(x8, w8, bias, outs, M, K, H, dk, alpha3, s);
    case 2: return run<G<128, 256, 64, Coop>>(x8, w8, bias, outs, M, K, H, dk, alpha3, s);
    case 3: return run<G<128, 128, 64, Ping>>(x8, w8, bias, outs, M, K, H, dk, alpha3, s);
    case 4: return run<G<128, 128, 64, Coop>>(x8, w8, bias, outs, M, K, H, dk, alpha3, s);
    case 5: return run<G<128, 64, 128, Ping>>(x8, w8, bias, outs, M, K, H, dk, alpha3, s);
    default: return run<G<128, 128, 128, Ping>>(x8, w8, bias, outs, M, K, H, dk, alpha3, s);
  }
}

}  // extern "C"

// ---------------------------------------------------------------------------------------------------------------
// TensorRT IPluginV3 "QKVHeads" (version 1). Input: FP8 [B, T, K] (TensorRT's quantise of the layer-normed residual).
// Outputs: q_u, k, v fp16 [H, B, T, dk]. Fields: w (1-byte e4m3 [3*H*dk*K]), bias (float32 [3*H*dk]),
// alpha (float32 [3] = s_x * s_w for q, k, v), dims (int32 [K, H, dk]). Emitted by lean_encoder (LEAN_QKV_PLUGIN=1).

#include <NvInferRuntime.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

extern "C" void* sparse_fp8_prepare_l(const void* W, int N, int K, int L, cudaStream_t s);
extern "C" int sparse_qkv_run(void* h3[3], const void* x8, const float* bias, void* outs[3], int M, int H, int dk,
                              const float* alpha3, cudaStream_t s);

namespace {

using namespace nvinfer1;

struct QKVWeights {
  std::vector<uint8_t> w;
  std::vector<float> bias;
  float alpha[3] = {1.f, 1.f, 1.f};
  int K = 0, H = 0, dk = 0;
  int32_t sparse24 = 0;               // weights 2:4 along K: CUTLASS sparse, compressed per head (sparse_fp8.cu)
  void* sh[3] = {nullptr, nullptr, nullptr};
  void* dw = nullptr;
  float* db = nullptr;
  ~QKVWeights() {
    if (dw) cudaFree(dw);
    if (db) cudaFree(db);
  }
  bool upload() {
    if (dw) return true;
    if (cudaMalloc(&dw, w.size()) != cudaSuccess || cudaMalloc(&db, bias.size() * sizeof(float)) != cudaSuccess)
      return false;
    if (cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(db, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
    if (sparse24)
      for (int p = 0; p < 3; ++p) {
        sh[p] = sparse_fp8_prepare_l(static_cast<uint8_t*>(dw) + static_cast<size_t>(p) * H * dk * K, dk, K, H, 0);
        if (!sh[p]) return false;
      }
    return true;
  }
};

class QKVHeads : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  explicit QKVHeads(std::shared_ptr<QKVWeights> w) : w_(std::move(w)) {
    dims_[0] = w_->K; dims_[1] = w_->H; dims_[2] = w_->dk;
    fields_ = {PluginField("w", w_->w.data(), PluginFieldType::kINT8, static_cast<int32_t>(w_->w.size())),
               PluginField("bias", w_->bias.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(w_->bias.size())),
               PluginField("alpha", w_->alpha, PluginFieldType::kFLOAT32, 3),
               PluginField("dims", dims_, PluginFieldType::kINT32, 3),
               PluginField("sparse24", &w_->sparse24, PluginFieldType::kINT32, 1)};
    fc_.nbFields = 5;
    fc_.fields = fields_.data();
  }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
    if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
    if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
  }
  IPluginV3* clone() noexcept override { return new QKVHeads(w_); }
  AsciiChar const* getPluginName() const noexcept override { return "QKVHeads"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) noexcept override { return w_->upload() ? 0 : -1; }
  int32_t getOutputDataTypes(DataType* out, int32_t n, DataType const*, int32_t) const noexcept override {
    for (int i = 0; i < n; ++i) out[i] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t n,
                          IExprBuilder& eb) noexcept override {
    for (int i = 0; i < n; ++i) {
      out[i].nbDims = 4;
      out[i].d[0] = eb.constant(w_->H);
      out[i].d[1] = in[0].d[0];
      out[i].d[2] = in[0].d[1];
      out[i].d[3] = eb.constant(w_->dk);
    }
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override {
    if (io[pos].desc.format != TensorFormat::kLINEAR) return false;
    return pos == 0 ? io[0].desc.type == DataType::kFP8 : io[pos].desc.type == DataType::kHALF;
  }
  int32_t getNbOutputs() const noexcept override { return 3; }
  int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    return w_->upload() ? 0 : -1;
  }
  int32_t enqueue(PluginTensorDesc const* id, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (!w_->upload()) return -1;
    if (id[0].dims.nbDims != 3 || id[0].dims.d[2] != w_->K) return -2;
    int M = static_cast<int>(id[0].dims.d[0] * id[0].dims.d[1]);
    if (M == 0) return 0;
    void* outs[3] = {outputs[0], outputs[1], outputs[2]};
    int r = w_->sparse24 ? sparse_qkv_run(w_->sh, inputs[0], w_->db, outs, M, w_->H, w_->dk, w_->alpha, stream)
                         : qkv_heads_run(inputs[0], w_->dw, w_->db, outputs[0], outputs[1], outputs[2], M, w_->K,
                                         w_->H, w_->dk, w_->alpha, stream);
    if (r) fprintf(stderr, "QKVHeads: kernel error %d (M=%d)\n", r, M);
    return r ? -3 : 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fc_; }

 private:
  std::shared_ptr<QKVWeights> w_;
  int32_t dims_[3];
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class QKVHeadsCreator : public IPluginCreatorV3One {
 public:
  QKVHeadsCreator() {
    attrs_ = {PluginField("w", nullptr, PluginFieldType::kINT8, 0), PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 0),
              PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 3), PluginField("dims", nullptr, PluginFieldType::kINT32, 3),
              PluginField("sparse24", nullptr, PluginFieldType::kINT32, 1)};
    fc_.nbFields = 5;
    fc_.fields = attrs_.data();
  }
  IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    auto w = std::make_shared<QKVWeights>();
    int have = 0;
    for (int i = 0; i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      std::string n = f.name;
      if (n == "w") {
        auto p = static_cast<uint8_t const*>(f.data);
        size_t bytes = f.length * (f.type == PluginFieldType::kINT8 || f.type == PluginFieldType::kCHAR ? 1 : 0);
        if (!bytes) { fprintf(stderr, "QKVHeads %s: w must be 1-byte data\n", name); return nullptr; }
        w->w.assign(p, p + bytes); ++have;
      } else if (n == "bias") {
        auto p = static_cast<float const*>(f.data); w->bias.assign(p, p + f.length); ++have;
      } else if (n == "alpha") {
        if (f.length != 3 || f.type != PluginFieldType::kFLOAT32) {
          fprintf(stderr, "QKVHeads %s: alpha must be 3 float32 (q, k, v)\n", name);
          return nullptr;
        }
        for (int j = 0; j < 3; ++j) w->alpha[j] = static_cast<float const*>(f.data)[j];
        ++have;
      } else if (n == "sparse24") {
        w->sparse24 = f.type == PluginFieldType::kINT64 ? static_cast<int32_t>(*static_cast<int64_t const*>(f.data))
                                                        : *static_cast<int32_t const*>(f.data);
      } else if (n == "dims") {
        if (f.type == PluginFieldType::kINT64) {
          auto d = static_cast<int64_t const*>(f.data); w->K = int(d[0]); w->H = int(d[1]); w->dk = int(d[2]);
        } else {
          auto d = static_cast<int32_t const*>(f.data); w->K = d[0]; w->H = d[1]; w->dk = d[2];
        }
        ++have;
      }
    }
    size_t N = 3u * w->H * w->dk;
    if (have != 4 || w->w.size() != N * w->K || w->bias.size() != N || (w->dk != 128 && w->dk != 64) ||
        (w->sparse24 && w->dk != 128)) {
      fprintf(stderr, "QKVHeads %s: bad fields (w %zu, bias %zu, K %d H %d dk %d)\n", name, w->w.size(),
              w->bias.size(), w->K, w->H, w->dk);
      return nullptr;
    }
    return new QKVHeads(w);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  AsciiChar const* getPluginName() const noexcept override { return "QKVHeads"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

 private:
  std::vector<PluginField> attrs_;
  PluginFieldCollection fc_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(QKVHeadsCreator);
