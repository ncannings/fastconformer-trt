// TensorRT IPluginV3 "PwGluFp8" (version 1): the conv module's first pointwise linear plus GLU in two CUTLASS FP8
// GEMMs (ffn_fp8.cu fp8_pw_glu_run): the gate half to an fp16 scratch, then the value half with
// y = value * sigmoid(gate) in its epilogue.
//
// Hypothesis: TensorRT writes pw1's [M, 2D] output (26 MB per 32 x 16 s batch per layer for D = 1024) and its fused
// GLU + depthwise kernel reads it back; writing the gate [M, D] once and the GLU output [M, D] instead saves about
// a quarter of that traffic, ~0.1 ms per layer. Same weights and FP8 inputs; the only change is where sigmoid and
// the product are computed (fp32 in the epilogue). Ablate by building without GLU_FUSE=1.
// Input 0: FP8 [.., K] (TensorRT's quantise). Output fp16 [.., N]. Fields: w (e4m3 bytes [2N*K], value rows then
// gate rows), dims int32 [K, N], alpha float32 (s_x * s_w). Inserted by glu_surgery.py. No biases.

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

extern "C" int fp8_pw_glu_run(const void* x8, const void* w, void* gate, void* y, int M, int N, int K, float alpha,
                              cudaStream_t s);

namespace {

struct GluW {
  std::vector<uint8_t> w;
  int K = 0, N = 0;
  float alpha = 1.f;
  void* dw = nullptr;
  ~GluW() { if (dw) cudaFree(dw); }
  bool upload() {
    if (dw) return true;
    return cudaMalloc(&dw, w.size()) == cudaSuccess &&
           cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice) == cudaSuccess;
  }
};

class PwGluFp8 : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  explicit PwGluFp8(std::shared_ptr<GluW> w) : w_(std::move(w)) {
    dims_[0] = w_->K;
    dims_[1] = w_->N;
    fields_ = {PluginField("w", w_->w.data(), PluginFieldType::kINT8, static_cast<int32_t>(w_->w.size())),
               PluginField("dims", dims_, PluginFieldType::kINT32, 2),
               PluginField("alpha", &w_->alpha, PluginFieldType::kFLOAT32, 1)};
    fc_.nbFields = 3;
    fc_.fields = fields_.data();
  }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
    if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
    if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
  }
  IPluginV3* clone() noexcept override { return new PwGluFp8(w_); }
  AsciiChar const* getPluginName() const noexcept override { return "PwGluFp8"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) noexcept override { return w_->upload() ? 0 : -1; }
  int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override {
    out[0] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t,
                          IExprBuilder& eb) noexcept override {
    out[0] = in[0];
    out[0].d[in[0].nbDims - 1] = eb.constant(w_->N);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override {
    if (io[pos].desc.format != TensorFormat::kLINEAR) return false;
    return pos == 0 ? io[0].desc.type == DataType::kFP8 : io[pos].desc.type == DataType::kHALF;
  }
  int32_t getNbOutputs() const noexcept override { return 1; }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) const noexcept override {
    int64_t M = 1;
    for (int i = 0; i < in[0].max.nbDims - 1; ++i) M *= in[0].max.d[i];
    return static_cast<size_t>(M) * w_->N * 2 + 256;            // fp16 gate
  }
  int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    return w_->upload() ? 0 : -1;
  }
  int32_t enqueue(PluginTensorDesc const* id, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
    if (!w_->upload()) return -1;
    int64_t M = 1;
    for (int i = 0; i < id[0].dims.nbDims - 1; ++i) M *= id[0].dims.d[i];
    if (id[0].dims.d[id[0].dims.nbDims - 1] != w_->K) return -2;
    if (M == 0) return 0;
    int r = fp8_pw_glu_run(inputs[0], w_->dw, workspace, outputs[0], static_cast<int>(M), w_->N, w_->K, w_->alpha,
                           stream);
    if (r) fprintf(stderr, "PwGluFp8: kernel error %d (M=%lld)\n", r, static_cast<long long>(M));
    return r ? -3 : 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fc_; }

 private:
  std::shared_ptr<GluW> w_;
  int32_t dims_[2];
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class PwGluFp8Creator : public IPluginCreatorV3One {
 public:
  PwGluFp8Creator() {
    attrs_ = {PluginField("w", nullptr, PluginFieldType::kINT8, 0), PluginField("dims", nullptr, PluginFieldType::kINT32, 2),
              PluginField("alpha", nullptr, PluginFieldType::kFLOAT32, 1)};
    fc_.nbFields = 3;
    fc_.fields = attrs_.data();
  }
  IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    auto w = std::make_shared<GluW>();
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
      } else if (n == "alpha") {
        w->alpha = f.type == PluginFieldType::kFLOAT64 ? static_cast<float>(*static_cast<double const*>(f.data))
                                                       : *static_cast<float const*>(f.data);
        ++have;
      }
    }
    if (have != 3 || w->w.size() != 2u * w->N * w->K || w->K % 128 || w->N % 128) {
      fprintf(stderr, "PwGluFp8 %s: bad fields (w %zu, K %d, N %d)\n", name, w->w.size(), w->K, w->N);
      return nullptr;
    }
    return new PwGluFp8(w);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  AsciiChar const* getPluginName() const noexcept override { return "PwGluFp8"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

 private:
  std::vector<PluginField> attrs_;
  PluginFieldCollection fc_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(PwGluFp8Creator);
