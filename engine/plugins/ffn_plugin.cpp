// TensorRT IPluginV3 "FFNFp8" (version 1): the encoder's whole feed-forward block in two CUTLASS kernels
// (ffn_fp8.cu). Input: the FP8 output of TensorRT's quantise of the layer-normed residual [.., K1]. Output: the FF
// block's fp16 result [.., N2] (before the 0.5 half-step and residual add, which stay in TensorRT). The FP8
// intermediate [M, N1] lives in the plugin workspace.
// Fields: w1 [N1*K1] and w2 [N2*N1] FP8 e4m3 bytes (any 1-byte field type), dims int32 [K1, N1, N2],
// alpha1 = s_x * s_w1, oscale = 1 / s_h, alpha2 = s_h * s_w2 (float32); optional res_scale > 0 (residual mode: a
// second input, the residual, and output = residual + res_scale * FF, fp32 or fp16 as the residual). ffn_surgery.py.
// Optional fp4 = 1 (with sx = s_x): NVFP4 feed-forward (nvfp4_ffn.cu, CUTLASS >= 4.8). Input 0 is then the fp16
// layer-normed residual (not TensorRT's FP8 quantise); the plugin quantises it with global scale s_x / 6, GEMM1
// writes SiLU(h) as NVFP4 (global scale s_h / 6), GEMM2 adds the fp16 residual. The FP8 weight bytes are decoded and
// re-quantised to NVFP4 once at upload (global scale amax / (6 * 448)). Accuracy trade: see PREREG_MERGE.md.
// No silent fallbacks: wrong field sizes or a failing kernel return an error to TensorRT.

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

extern "C" int ffn_fp8_run_res(const void* x8, void* h8, const void* res, void* y, const void* w1, const void* w2,
                               int M, int K1, int N1, int N2, float alpha1, float oscale, float alpha2, float res_scale,
                               int is_f32, cudaStream_t s);
extern "C" void* sparse_fp8_prepare(const void* W, int N, int K, cudaStream_t s);
extern "C" int sparse_ffn_run(void* h1, void* h2, const void* x8, void* h8, const void* res, void* y, int M,
                              float alpha1, float oscale, float alpha2, float res_scale, int is_f32, cudaStream_t s);
extern "C" int ffn_fp8_run_bias(const void* x8, void* h8, const void* res, void* y, const void* w1, const void* w2,
                                const float* b1, const float* b2s, int M, int K1, int N1, int N2, float alpha1,
                                float oscale, float alpha2s, int is_f32, cudaStream_t s);
extern "C" size_t nvfp4_sf_bytes(int R, int K);
extern "C" int nvfp4_quant(const void* x, void* q, void* sf, int R, int K, float g, int as_b, cudaStream_t s);
extern "C" int nvfp4_gemm1(const void* xq, const void* sfx, const void* wq, const void* sfw, void* hq, void* sfh, int M,
                           int N, int K, float alpha, const float* norm_constant, cudaStream_t s);
extern "C" int nvfp4_gemm2(const void* hq, const void* sfh, const void* wq, const void* sfw, const void* res, void* y,
                           int M, int N, int K, float alpha, float beta, cudaStream_t s);
extern "C" int ffn_fp8_run(const void* x8, void* h8, void* y, const void* w1, const void* w2, int M, int K1, int N1,
                           int N2, float alpha1, float oscale, float alpha2, void* ws, size_t ws_bytes,
                           cudaStream_t s);

namespace {

constexpr char const* kName = "FFNFp8";
constexpr char const* kVersion = "1";

struct Weights {
  std::vector<uint8_t> w1, w2;  // host copies (serialised with the engine)
  std::vector<float> b1, b2;    // optional biases (empty = none); b2 is pre-scaled by res_scale on the device
  float* db = nullptr;          // device b1 | b2 * scale
  int K1 = 0, N1 = 0, N2 = 0;
  float alpha1 = 1.f, oscale = 1.f, alpha2 = 1.f;
  float res_scale = 0.f;   // > 0: residual mode, second input is the residual, y = residual + res_scale * FF(x)
  int32_t sparse24 = 0;    // 1: weights are 2:4 sparse along K; run CUTLASS sparse kernels (sparse_fp8.cu)
  void* sh1 = nullptr;     // compressed sparse operands (sparse_fp8_prepare)
  void* sh2 = nullptr;
  void* d1 = nullptr;
  void* d2 = nullptr;
  int32_t fp4 = 0;         // 1: NVFP4 path (input fp16; weights re-quantised at upload)
  float sx = 0.f;          // FP8 input scale s_x (fp4 needs it separately from alpha1)
  void *q1 = nullptr, *f1 = nullptr, *q2 = nullptr, *f2 = nullptr;   // NVFP4 weights + scale factors
  float g1 = 0.f, g2 = 0.f, gx = 0.f, gh = 0.f;
  float* dnorm = nullptr;  // device 1 / gh (GEMM1 output global scale)
  ~Weights() {
    for (void* p : {d1, d2, static_cast<void*>(db), q1, f1, q2, f2, static_cast<void*>(dnorm)})
      if (p) cudaFree(p);
  }
  static float e4m3(uint8_t b) {
    int sgn = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7);
    return sgn ? -v : v;
  }
  bool to_fp4(std::vector<uint8_t> const& w8, float s_w, int R, int K, void** q, void** f, float* g) {
    std::vector<__half> h(w8.size());
    float amax = 0.f;
    for (size_t i = 0; i < w8.size(); ++i) {
      float v = e4m3(w8[i]) * s_w;
      h[i] = __float2half_rn(v);
      amax = std::max(amax, std::fabs(v));
    }
    *g = amax > 0.f ? amax / (6.f * 448.f) : 1.f;
    void* dh = nullptr;
    bool ok = cudaMalloc(&dh, h.size() * 2) == cudaSuccess &&
              cudaMemcpy(dh, h.data(), h.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMalloc(q, static_cast<size_t>(R) * K / 2) == cudaSuccess &&
              cudaMalloc(f, nvfp4_sf_bytes(R, K)) == cudaSuccess &&
              nvfp4_quant(dh, *q, *f, R, K, *g, 1, 0) == 0 && cudaDeviceSynchronize() == cudaSuccess;
    if (dh) cudaFree(dh);
    return ok;
  }
  bool upload() {
    if (d1 || q1) return true;
    if (fp4) {
      const float s_w1 = alpha1 / sx, s_h = 1.f / oscale, s_w2 = alpha2 / s_h;
      gx = sx / 6.f;
      gh = s_h / 6.f;
      const float inv = 1.f / gh;
      return to_fp4(w1, s_w1, N1, K1, &q1, &f1, &g1) && to_fp4(w2, s_w2, N2, N1, &q2, &f2, &g2) &&
             cudaMalloc(&dnorm, sizeof(float)) == cudaSuccess &&
             cudaMemcpy(dnorm, &inv, sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (cudaMalloc(&d1, w1.size()) != cudaSuccess || cudaMalloc(&d2, w2.size()) != cudaSuccess) return false;
    if (!b1.empty()) {
      std::vector<float> h(b1);
      float sc = res_scale > 0.f ? res_scale : 1.f;
      for (float v : b2) h.push_back(v * sc);
      if (cudaMalloc(&db, h.size() * sizeof(float)) != cudaSuccess ||
          cudaMemcpy(db, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    }
    if (cudaMemcpy(d1, w1.data(), w1.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d2, w2.data(), w2.size(), cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
    if (sparse24) {  // compress after the weights are on the device
      sh1 = sparse_fp8_prepare(d1, N1, K1, 0);
      sh2 = sparse_fp8_prepare(d2, N2, N1, 0);
      if (!sh1 || !sh2) return false;
    }
    return true;
  }
};

class FFNFp8 : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  explicit FFNFp8(std::shared_ptr<Weights> w) : w_(std::move(w)) { buildFields(); }

  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
    if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
  }
  IPluginV3* clone() noexcept override { return new FFNFp8(w_); }

  AsciiChar const* getPluginName() const noexcept override { return kName; }
  AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) noexcept override {
    return w_->upload() ? 0 : -1;
  }
  int32_t getOutputDataTypes(DataType* out, int32_t, DataType const* in, int32_t nin) const noexcept override {
    out[0] = (w_->res_scale > 0.f && nin > 1) ? in[1] : DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t,
                          IExprBuilder& eb) noexcept override {
    out[0] = in[0];
    out[0].d[in[0].nbDims - 1] = eb.constant(w_->N2);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t nin, int32_t) noexcept override {
    if (io[pos].desc.format != TensorFormat::kLINEAR) return false;
    if (w_->fp4) return io[pos].desc.type == DataType::kHALF;   // fp16 input, fp16 residual and output
    if (pos == 0) return io[0].desc.type == DataType::kFP8;
    if (nin == 1) return io[pos].desc.type == DataType::kHALF;
    // residual fp32 or fp16 (TensorRT picks fp16, which is much faster: encoder 89.9 vs 103.9 ms per batch). WER is
    // unchanged on 2,000 Earnings-22 / dev-clean utterances (10.56 / 1.95 vs fp32-only 10.52 / 1.95, no fold 10.58 /
    // 1.95). FFN_RES_F32_ONLY restricts it for ablation.
#ifdef FFN_RES_F32_ONLY
    if (pos == 1) return io[1].desc.type == DataType::kFLOAT;
#else
    if (pos == 1) return io[1].desc.type == DataType::kFLOAT || io[1].desc.type == DataType::kHALF;
#endif
    return io[2].desc.type == io[1].desc.type;            // output matches the residual
  }
  int32_t getNbOutputs() const noexcept override { return 1; }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) const noexcept override {
    int64_t M = 1;
    for (int i = 0; i < in[0].max.nbDims - 1; ++i) M *= in[0].max.d[i];
    if (w_->fp4)
      return static_cast<size_t>(M) * (w_->K1 + w_->N1) / 2 + nvfp4_sf_bytes(static_cast<int>(M), w_->K1) +
             nvfp4_sf_bytes(static_cast<int>(M), w_->N1) + 1024;
    return static_cast<size_t>(M) * w_->N1 + 256;
  }

  int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    return w_->upload() ? 0 : -1;
  }
  int32_t enqueue(PluginTensorDesc const* id, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
    if (!w_->upload()) return -1;
    int64_t M = 1;
    for (int i = 0; i < id[0].dims.nbDims - 1; ++i) M *= id[0].dims.d[i];
    if (id[0].dims.d[id[0].dims.nbDims - 1] != w_->K1) return -2;
    if (M == 0) return 0;
    if (w_->fp4) {
      const int m = static_cast<int>(M);
      auto al = [](size_t v) { return (v + 255) / 256 * 256; };
      uint8_t* ws = static_cast<uint8_t*>(workspace);
      uint8_t* xq = ws;
      uint8_t* sfx = xq + al(static_cast<size_t>(m) * w_->K1 / 2);
      uint8_t* hq = sfx + al(nvfp4_sf_bytes(m, w_->K1));
      uint8_t* sfh = hq + al(static_cast<size_t>(m) * w_->N1 / 2);
      const bool res = w_->res_scale > 0.f;
      int r = nvfp4_quant(inputs[0], xq, sfx, m, w_->K1, w_->gx, 0, stream);
      if (!r) r = 10 * nvfp4_gemm1(xq, sfx, w_->q1, w_->f1, hq, sfh, m, w_->N1, w_->K1, w_->gx * w_->g1, w_->dnorm, stream);
      if (!r) r = 100 * nvfp4_gemm2(hq, sfh, w_->q2, w_->f2, res ? inputs[1] : nullptr, outputs[0], m, w_->N2, w_->N1,
                                    (res ? w_->res_scale : 1.f) * w_->gh * w_->g2, 1.f, stream);
      if (r) fprintf(stderr, "FFNFp8 (fp4): kernel error %d (M=%d)\n", r, m);
      return r ? -3 : 0;
    }
    int r = w_->sparse24
                ? sparse_ffn_run(w_->sh1, w_->sh2, inputs[0], workspace, w_->res_scale > 0.f ? inputs[1] : nullptr,
                                 outputs[0], static_cast<int>(M), w_->alpha1, w_->oscale, w_->alpha2, w_->res_scale,
                                 w_->res_scale > 0.f && id[1].type == DataType::kFLOAT, stream)
            : !w_->b1.empty()
                ? ffn_fp8_run_bias(inputs[0], workspace, w_->res_scale > 0.f ? inputs[1] : nullptr, outputs[0],
                                   w_->d1, w_->d2, w_->db, w_->db + w_->N1, static_cast<int>(M), w_->K1, w_->N1,
                                   w_->N2, w_->alpha1, w_->oscale,
                                   w_->alpha2 * (w_->res_scale > 0.f ? w_->res_scale : 1.f),
                                   w_->res_scale > 0.f && id[1].type == DataType::kFLOAT, stream)
            : w_->res_scale > 0.f
                ? ffn_fp8_run_res(inputs[0], workspace, inputs[1], outputs[0], w_->d1, w_->d2, static_cast<int>(M),
                                  w_->K1, w_->N1, w_->N2, w_->alpha1, w_->oscale, w_->alpha2, w_->res_scale,
                                  id[1].type == DataType::kFLOAT, stream)
                : ffn_fp8_run(inputs[0], workspace, outputs[0], w_->d1, w_->d2, static_cast<int>(M), w_->K1, w_->N1,
                              w_->N2, w_->alpha1, w_->oscale, w_->alpha2, nullptr, 0, stream);
    if (r) fprintf(stderr, "FFNFp8: kernel error %d (M=%lld)\n", r, static_cast<long long>(M));
    return r ? -3 : 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fc_; }

 private:
  void buildFields() {
    dims_[0] = w_->K1;
    dims_[1] = w_->N1;
    dims_[2] = w_->N2;
    fields_ = {PluginField("w1", w_->w1.data(), PluginFieldType::kINT8, static_cast<int32_t>(w_->w1.size())),
               PluginField("w2", w_->w2.data(), PluginFieldType::kINT8, static_cast<int32_t>(w_->w2.size())),
               PluginField("dims", dims_, PluginFieldType::kINT32, 3),
               PluginField("alpha1", &w_->alpha1, PluginFieldType::kFLOAT32, 1),
               PluginField("oscale", &w_->oscale, PluginFieldType::kFLOAT32, 1),
               PluginField("alpha2", &w_->alpha2, PluginFieldType::kFLOAT32, 1),
               PluginField("res_scale", &w_->res_scale, PluginFieldType::kFLOAT32, 1),
               PluginField("b1", w_->b1.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(w_->b1.size())),
               PluginField("b2", w_->b2.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(w_->b2.size())),
               PluginField("sparse24", &w_->sparse24, PluginFieldType::kINT32, 1),
               PluginField("fp4", &w_->fp4, PluginFieldType::kINT32, 1),
               PluginField("sx", &w_->sx, PluginFieldType::kFLOAT32, 1)};
    fc_.nbFields = static_cast<int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }
  std::shared_ptr<Weights> w_;
  int32_t dims_[3];
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

size_t typeBytes(PluginFieldType t) {
  switch (t) {
    case PluginFieldType::kFLOAT64: case PluginFieldType::kINT64: return 8;
    case PluginFieldType::kFLOAT32: case PluginFieldType::kINT32: return 4;
    case PluginFieldType::kFLOAT16: case PluginFieldType::kINT16: case PluginFieldType::kBF16: return 2;
    default: return 1;
  }
}

float asFloat(PluginField const& f) {
  if (f.type == PluginFieldType::kFLOAT64) return static_cast<float>(*static_cast<double const*>(f.data));
  return *static_cast<float const*>(f.data);
}

class FFNFp8Creator : public IPluginCreatorV3One {
 public:
  FFNFp8Creator() {
    attrs_ = {PluginField("w1", nullptr, PluginFieldType::kINT8, 0), PluginField("w2", nullptr, PluginFieldType::kINT8, 0),
              PluginField("dims", nullptr, PluginFieldType::kINT32, 3),
              PluginField("alpha1", nullptr, PluginFieldType::kFLOAT32, 1),
              PluginField("oscale", nullptr, PluginFieldType::kFLOAT32, 1),
              PluginField("alpha2", nullptr, PluginFieldType::kFLOAT32, 1),
              PluginField("res_scale", nullptr, PluginFieldType::kFLOAT32, 1),
              PluginField("b1", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("b2", nullptr, PluginFieldType::kFLOAT32, 0),
              PluginField("sparse24", nullptr, PluginFieldType::kINT32, 1),
              PluginField("fp4", nullptr, PluginFieldType::kINT32, 1), PluginField("sx", nullptr, PluginFieldType::kFLOAT32, 1)};
    fc_.nbFields = static_cast<int32_t>(attrs_.size());
    fc_.fields = attrs_.data();
  }
  IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    auto w = std::make_shared<Weights>();
    bool have[6] = {};
    for (int i = 0; i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      std::string n = f.name;
      size_t bytes = static_cast<size_t>(f.length) * typeBytes(f.type);
      if (n == "w1" || n == "w2") {
        if (typeBytes(f.type) != 1) {
          fprintf(stderr, "FFNFp8 %s: field %s has a %zu-byte type, need 1-byte FP8 data\n", name, n.c_str(),
                  typeBytes(f.type));
          return nullptr;
        }
        auto& dst = n == "w1" ? w->w1 : w->w2;
        dst.assign(static_cast<uint8_t const*>(f.data), static_cast<uint8_t const*>(f.data) + bytes);
        have[n == "w1" ? 0 : 1] = true;
      } else if (n == "dims") {
        if (f.type == PluginFieldType::kINT64) {
          auto d = static_cast<int64_t const*>(f.data);
          w->K1 = static_cast<int>(d[0]); w->N1 = static_cast<int>(d[1]); w->N2 = static_cast<int>(d[2]);
        } else {
          auto d = static_cast<int32_t const*>(f.data);
          w->K1 = d[0]; w->N1 = d[1]; w->N2 = d[2];
        }
        have[2] = true;
      } else if (n == "alpha1") { w->alpha1 = asFloat(f); have[3] = true; }
      else if (n == "oscale") { w->oscale = asFloat(f); have[4] = true; }
      else if (n == "alpha2") { w->alpha2 = asFloat(f); have[5] = true; }
      else if (n == "res_scale") { w->res_scale = asFloat(f); }
      else if (n == "sx") { w->sx = asFloat(f); }
      else if (n == "fp4") {
        w->fp4 = f.type == PluginFieldType::kINT64 ? static_cast<int32_t>(*static_cast<int64_t const*>(f.data))
                                                   : *static_cast<int32_t const*>(f.data);
      }
      else if (n == "sparse24") {
        w->sparse24 = f.type == PluginFieldType::kINT64 ? static_cast<int32_t>(*static_cast<int64_t const*>(f.data))
                                                        : *static_cast<int32_t const*>(f.data);
      }
      else if (n == "b1" || n == "b2") {
        auto p = static_cast<float const*>(f.data);
        (n == "b1" ? w->b1 : w->b2).assign(p, p + f.length);
      }
    }
    for (bool h : have)
      if (!h) { fprintf(stderr, "FFNFp8 %s: missing field\n", name); return nullptr; }
    if (w->fp4 && (w->sparse24 || !w->b1.empty() || w->sx <= 0.f || w->K1 % 32 || w->N1 % 32)) {
      fprintf(stderr, "FFNFp8 %s: fp4 needs sx > 0, no biases, no sparse24, K1/N1 multiples of 32\n", name);
      return nullptr;
    }
    if (w->sparse24 && !w->b1.empty()) {
      fprintf(stderr, "FFNFp8 %s: sparse24 with biases is not supported\n", name);
      return nullptr;
    }
    if (w->b1.empty() != w->b2.empty() || (!w->b1.empty() && (w->b1.size() != static_cast<size_t>(w->N1) ||
                                                             w->b2.size() != static_cast<size_t>(w->N2)))) {
      fprintf(stderr, "FFNFp8 %s: biases must be both absent or sized N1 and N2\n", name);
      return nullptr;
    }
    if (w->w1.size() != static_cast<size_t>(w->N1) * w->K1 || w->w2.size() != static_cast<size_t>(w->N2) * w->N1) {
      fprintf(stderr, "FFNFp8 %s: weight sizes %zu, %zu do not match dims %d %d %d\n", name, w->w1.size(),
              w->w2.size(), w->K1, w->N1, w->N2);
      return nullptr;
    }
    return new FFNFp8(w);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  AsciiChar const* getPluginName() const noexcept override { return kName; }
  AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

 private:
  std::vector<PluginField> attrs_;
  PluginFieldCollection fc_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(FFNFp8Creator);
