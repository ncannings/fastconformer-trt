// TensorRT IPluginV3 "RelPosAttn" (version 1): fused relative-position attention (relpos_attn.py, Triton), run from
// its compiled cubin through the CUDA driver API. Replaces TensorRT's position-score matmul, relative shift, attention
// and layout copies with one kernel (see relpos_attn.py for the hypothesis and the maths).
// Inputs: q_u, k, v [H, B, T, dk] fp16; p [H, 2T-1, dk] fp16; c [H, 2T-1] fp32; lengths [B] int32.
// Output: [B, T, H*dk] fp16. Fields: cubin (bytes), kname (string), smem, warps, bm, dk (int32; dk optional, 128).
// The kernel ABI (relpos_cubin.py): Q, K, V, P, C, LEN, O, S pointers; T, R, B, H u32; scale f32; global scratch ptr.

#include <NvInferRuntime.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

struct RelPosKernel {
  std::vector<uint8_t> cubin;
  std::string name;
  int32_t smem = 0, warps = 4, bm = 32, dk = 128;
  CUmodule mod = nullptr;
  CUfunction fn = nullptr;
  std::mutex mu;
  bool load() {
    std::lock_guard<std::mutex> g(mu);
    if (fn) return true;
    if (cuModuleLoadData(&mod, cubin.data()) != CUDA_SUCCESS) return false;
    if (cuModuleGetFunction(&fn, mod, name.c_str()) != CUDA_SUCCESS) return false;
    if (smem > 48 * 1024)
      cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem);
    return true;
  }
};

class RelPosAttn : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  explicit RelPosAttn(std::shared_ptr<RelPosKernel> k) : k_(std::move(k)) {
    fields_ = {PluginField("cubin", k_->cubin.data(), PluginFieldType::kINT8, static_cast<int32_t>(k_->cubin.size())),
               PluginField("kname", k_->name.c_str(), PluginFieldType::kCHAR, static_cast<int32_t>(k_->name.size())),
               PluginField("smem", &k_->smem, PluginFieldType::kINT32, 1),
               PluginField("warps", &k_->warps, PluginFieldType::kINT32, 1),
               PluginField("bm", &k_->bm, PluginFieldType::kINT32, 1),
               PluginField("dk", &k_->dk, PluginFieldType::kINT32, 1)};
    fc_.nbFields = static_cast<int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
    if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
    if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
  }
  IPluginV3* clone() noexcept override { return new RelPosAttn(k_); }
  AsciiChar const* getPluginName() const noexcept override { return "RelPosAttn"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) noexcept override { return 0; }
  int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override {
    out[0] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t,
                          IExprBuilder& eb) noexcept override {
    out[0].nbDims = 3;
    out[0].d[0] = in[0].d[1];
    out[0].d[1] = in[0].d[2];
    out[0].d[2] = eb.operation(DimensionOperation::kPROD, *in[0].d[0], *in[0].d[3]);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override {
    if (io[pos].desc.format != TensorFormat::kLINEAR) return false;
    DataType t = io[pos].desc.type;
    switch (pos) {
      case 4: return t == DataType::kFLOAT;
      case 5: return t == DataType::kINT32;
      default: return t == DataType::kHALF;   // q_u, k, v, p, output
    }
  }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* id, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (!k_->load()) { fprintf(stderr, "RelPosAttn: cubin load failed\n"); return -1; }
    uint32_t H = id[0].dims.d[0], B = id[0].dims.d[1], T = id[0].dims.d[2], DK = id[0].dims.d[3];
    uint32_t R = id[3].dims.d[1];
    if (DK != static_cast<uint32_t>(k_->dk) || id[3].dims.d[0] != static_cast<int64_t>(H) || R != 2 * T - 1) return -2;
    if (B == 0 || T == 0) return 0;
    float scale = 1.f / std::sqrt(static_cast<float>(DK));
    CUdeviceptr q = (CUdeviceptr)inputs[0], k = (CUdeviceptr)inputs[1], v = (CUdeviceptr)inputs[2];
    CUdeviceptr p = (CUdeviceptr)inputs[3], c = (CUdeviceptr)inputs[4], len = (CUdeviceptr)inputs[5];
    CUdeviceptr o = (CUdeviceptr)outputs[0], s = o, gs = 0;   // S unused by the gather variant
    void* args[] = {&q, &k, &v, &p, &c, &len, &o, &s, &T, &R, &B, &H, &scale, &gs};
    unsigned gx = (T + k_->bm - 1) / k_->bm, gy = H * B;
    CUresult r = cuLaunchKernel(k_->fn, gx, gy, 1, 32 * k_->warps, 1, 1, k_->smem, stream, args, nullptr);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "RelPosAttn: launch error %d\n", static_cast<int>(r)); return -3; }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fc_; }

 private:
  std::shared_ptr<RelPosKernel> k_;
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class RelPosAttnCreator : public IPluginCreatorV3One {
 public:
  RelPosAttnCreator() {
    attrs_ = {PluginField("cubin", nullptr, PluginFieldType::kINT8, 0), PluginField("kname", nullptr, PluginFieldType::kCHAR, 0),
              PluginField("smem", nullptr, PluginFieldType::kINT32, 1), PluginField("warps", nullptr, PluginFieldType::kINT32, 1),
              PluginField("bm", nullptr, PluginFieldType::kINT32, 1), PluginField("dk", nullptr, PluginFieldType::kINT32, 1)};
    fc_.nbFields = static_cast<int32_t>(attrs_.size());
    fc_.fields = attrs_.data();
  }
  IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    auto k = std::make_shared<RelPosKernel>();
    int have = 0;
    auto i32 = [](PluginField const& f) {
      return f.type == PluginFieldType::kINT64 ? static_cast<int32_t>(*static_cast<int64_t const*>(f.data))
                                               : *static_cast<int32_t const*>(f.data);
    };
    for (int i = 0; i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      std::string n = f.name;
      if (n == "cubin") {
        auto p = static_cast<uint8_t const*>(f.data);
        k->cubin.assign(p, p + f.length); ++have;
      } else if (n == "kname") {
        k->name.assign(static_cast<char const*>(f.data), f.length);
        while (!k->name.empty() && k->name.back() == '\0') k->name.pop_back();
        ++have;
      } else if (n == "smem") { k->smem = i32(f); ++have; }
      else if (n == "warps") { k->warps = i32(f); ++have; }
      else if (n == "bm") { k->bm = i32(f); ++have; }
      else if (n == "dk") k->dk = i32(f);   // optional: engines built before head-dim 64 support are 128
    }
    if (have != 5 || k->cubin.empty() || k->name.empty()) {
      fprintf(stderr, "RelPosAttn %s: missing fields\n", name);
      return nullptr;
    }
    return new RelPosAttn(k);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  AsciiChar const* getPluginName() const noexcept override { return "RelPosAttn"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

 private:
  std::vector<PluginField> attrs_;
  PluginFieldCollection fc_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(RelPosAttnCreator);
