// TensorRT IPluginV3 "SpLinearFp8" (version 1): a 2:4-sparse FP8 linear (CUTLASS sm120 sparse, sparse_fp8.cu) for
// the encoder's remaining GEMMs once the model is pruned (conv-module pointwise convs, attention output).
// Input 0: FP8 [.., K] (TensorRT's quantise). Optional input 1: residual [.., N] (fp32 or fp16); then
// output = residual + res_scale * alpha * x @ W^T in the residual's type, else fp16 alpha * x @ W^T.
// Fields: w (e4m3 bytes [N*K], 2:4 along K), dims int32 [K, N], alpha, res_scale (float32). Inserted by
// sparse_surgery.py only after it has checked the 2:4 pattern. Optional dense (int32) = 1: plain FP8 weights, dense
// CUTLASS GEMM with the same residual epilogue (SP_DENSE=1 surgery on an unpruned model; folds the residual add).

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

extern "C" void* sparse_fp8_prepare(const void* W, int N, int K, cudaStream_t s);
extern "C" int sparse_linear_run(void* h, const void* x8, const void* res, void* y, int M, float alpha,
                                 float res_scale, int is_f32, cudaStream_t s);
extern "C" int fp8_linear_resb_run(const void* x8, const void* w, const float* bs, const void* res, void* y, int M,
                                   int N, int K, float alpha, float res_scale, int is_f32, cudaStream_t s);
extern "C" int fp8_linear_res_run(const void* x8, const void* w, const void* res, void* y, int M, int N, int K,
                                  float alpha, float res_scale, int is_f32, cudaStream_t s);

namespace {

struct SpW {
  std::vector<uint8_t> w;
  int K = 0, N = 0;
  float alpha = 1.f, res_scale = 0.f;
  int32_t dense = 0;
  std::vector<float> bias;   // optional (dense only): fp32 [N]
  float* db = nullptr;       // device bias * res_scale
  void* dw = nullptr;
  void* h = nullptr;
  ~SpW() {
    if (dw) cudaFree(dw);
    if (db) cudaFree(db);
  }
  bool upload() {
    if (h) return true;
    if (cudaMalloc(&dw, w.size()) != cudaSuccess ||
        cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
    if (!bias.empty()) {
      std::vector<float> b(bias);
      for (float& v : b) v *= res_scale;
      if (cudaMalloc(&db, b.size() * sizeof(float)) != cudaSuccess ||
          cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    }
    h = dense ? dw : sparse_fp8_prepare(dw, N, K, 0);
    return h != nullptr;
  }
};

class SpLinearFp8 : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  explicit SpLinearFp8(std::shared_ptr<SpW> w) : w_(std::move(w)) {
    dims_[0] = w_->K;
    dims_[1] = w_->N;
    fields_ = {PluginField("w", w_->w.data(), PluginFieldType::kINT8, static_cast<int32_t>(w_->w.size())),
               PluginField("dims", dims_, PluginFieldType::kINT32, 2),
               PluginField("alpha", &w_->alpha, PluginFieldType::kFLOAT32, 1),
               PluginField("res_scale", &w_->res_scale, PluginFieldType::kFLOAT32, 1),
               PluginField("dense", &w_->dense, PluginFieldType::kINT32, 1),
               PluginField("bias", w_->bias.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(w_->bias.size()))};
    fc_.nbFields = w_->bias.empty() ? 5 : 6;
    fc_.fields = fields_.data();
  }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
    if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
    if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
  }
  IPluginV3* clone() noexcept override { return new SpLinearFp8(w_); }
  AsciiChar const* getPluginName() const noexcept override { return "SpLinearFp8"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) noexcept override { return w_->upload() ? 0 : -1; }
  int32_t getOutputDataTypes(DataType* out, int32_t, DataType const* in, int32_t nin) const noexcept override {
    out[0] = nin > 1 ? in[1] : DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t,
                          IExprBuilder& eb) noexcept override {
    out[0] = in[0];
    out[0].d[in[0].nbDims - 1] = eb.constant(w_->N);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t nin, int32_t) noexcept override {
    if (io[pos].desc.format != TensorFormat::kLINEAR) return false;
    if (pos == 0) return io[0].desc.type == DataType::kFP8;
    if (nin == 1) return io[pos].desc.type == DataType::kHALF;
    if (pos == 1) return io[1].desc.type == DataType::kFLOAT || io[1].desc.type == DataType::kHALF;
    return io[2].desc.type == io[1].desc.type;
  }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    return w_->upload() ? 0 : -1;
  }
  int32_t enqueue(PluginTensorDesc const* id, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (!w_->upload()) return -1;
    int64_t M = 1;
    for (int i = 0; i < id[0].dims.nbDims - 1; ++i) M *= id[0].dims.d[i];
    if (M == 0) return 0;
    bool res = w_->res_scale > 0.f;
    int r = w_->db
                ? fp8_linear_resb_run(inputs[0], w_->dw, w_->db, inputs[1], outputs[0], static_cast<int>(M), w_->N, w_->K,
                                      w_->alpha, w_->res_scale, id[1].type == DataType::kFLOAT, stream)
            : w_->dense
                ? fp8_linear_res_run(inputs[0], w_->dw, inputs[1], outputs[0], static_cast<int>(M), w_->N, w_->K,
                                     w_->alpha, w_->res_scale, id[1].type == DataType::kFLOAT, stream)
                : sparse_linear_run(w_->h, inputs[0], res ? inputs[1] : nullptr, outputs[0], static_cast<int>(M),
                                    w_->alpha, w_->res_scale, res && id[1].type == DataType::kFLOAT, stream);
    if (r) fprintf(stderr, "SpLinearFp8: kernel error %d (M=%lld)\n", r, static_cast<long long>(M));
    return r ? -3 : 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fc_; }

 private:
  std::shared_ptr<SpW> w_;
  int32_t dims_[2];
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class SpLinearFp8Creator : public IPluginCreatorV3One {
 public:
  SpLinearFp8Creator() {
    attrs_ = {PluginField("w", nullptr, PluginFieldType::kINT8, 0), PluginField("dims", nullptr, PluginFieldType::kINT32, 2),
              PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1),
              PluginField("res_scale", nullptr, PluginFieldType::kFLOAT32, 1),
              PluginField("dense", nullptr, PluginFieldType::kINT32, 1), PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 0)};
    fc_.nbFields = 6;
    fc_.fields = attrs_.data();
  }
  IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    auto w = std::make_shared<SpW>();
    int have = 0;
    for (int i = 0; i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      std::string n = f.name;
      if (n == "w") {
        auto p = static_cast<uint8_t const*>(f.data);
        w->w.assign(p, p + f.length); ++have;
      } else if (n == "dims") {
        if (f.type == PluginFieldType::kINT64) {
          auto d = static_cast<int64_t const*>(f.data); w->K = int(d[0]); w->N = int(d[1]);
        } else {
          auto d = static_cast<int32_t const*>(f.data); w->K = d[0]; w->N = d[1];
        }
        ++have;
      } else if (n == "alpha" || n == "res_scale") {
        float v = f.type == PluginFieldType::kFLOAT64 ? static_cast<float>(*static_cast<double const*>(f.data))
                                                      : *static_cast<float const*>(f.data);
        (n == "alpha" ? w->alpha : w->res_scale) = v;
        have += n == "alpha";
      } else if (n == "bias") {
        auto p = static_cast<float const*>(f.data);
        w->bias.assign(p, p + f.length);
      } else if (n == "dense") {
        w->dense = f.type == PluginFieldType::kINT64 ? static_cast<int32_t>(*static_cast<int64_t const*>(f.data))
                                                     : *static_cast<int32_t const*>(f.data);
      }
    }
    if (!w->bias.empty() && (!w->dense || w->bias.size() != static_cast<size_t>(w->N))) {
      fprintf(stderr, "SpLinearFp8 %s: bias needs dense mode and N values\n", name);
      return nullptr;
    }
    if (w->dense && w->res_scale <= 0.f) {
      fprintf(stderr, "SpLinearFp8 %s: dense mode needs the residual (res_scale > 0)\n", name);
      return nullptr;
    }
    if (have != 3 || w->w.size() != static_cast<size_t>(w->N) * w->K || w->K % 128 || w->N % 128) {
      fprintf(stderr, "SpLinearFp8 %s: bad fields (w %zu, K %d, N %d)\n", name, w->w.size(), w->K, w->N);
      return nullptr;
    }
    return new SpLinearFp8(w);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  AsciiChar const* getPluginName() const noexcept override { return "SpLinearFp8"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

 private:
  std::vector<PluginField> attrs_;
  PluginFieldCollection fc_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(SpLinearFp8Creator);
