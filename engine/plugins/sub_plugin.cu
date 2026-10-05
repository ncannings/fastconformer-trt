// TensorRT IPluginV3 "SubConv02" (version 1): the first two convolutions of FastConformer's dw_striding subsampling
// (conv.0: 1 -> C, 3x3, stride 2, pad 1, + ReLU; conv.2: depthwise C, 3x3, stride 2, pad 1) in one kernel.
//
// Hypothesis: TensorRT writes conv.0's output ([B, 256, T/2, 64], 840 MB fp16 per 32 x 16 s batch) and reads it
// straight back for the depthwise conv: about 9 ms of pure memory traffic per batch on GB10. Each output of
// conv.2 depends on a 7x7 input window, so computing both from a shared-memory input tile never materialises the
// intermediate. Same weights and arithmetic (fp32 accumulation), so it should match TensorRT to fp16 rounding.
// Ablate by building from the ONNX without sub_surgery.py.
//
// masked (int32, optional): second input lengths int32 [B]; rows past each stage's length are zeroed as NeMo's
// MaskedConvSequential does (needed where the subsampling convs have biases that leak into padding; nhwc only).
// Input [B, 1, T, F] fp16, output [B, C, T2, F2] fp16 (linear, or channels-last HWC8 when TensorRT asks for it), T1 = (T - 1) / 2 + 1, T2 = (T1 - 1) / 2 + 1.
// Fields: w0 [C*9], b0 [C], w2 [C*9], b2 [C] (float32); optional nhwc (int32): output [B, T2, F2, C] linear (the
// layout TensorRT's following 1x1-conv GEMM uses; sub_surgery.py adds a Transpose back to NCHW that TensorRT folds
// into its own layout change, removing a 2 ms Move per batch).

#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace subconv {
using namespace nvcuda;

constexpr int TX = 32, TY = 8, CPB = 32;               // f2 x t2 outputs per block, channels per block
constexpr int SH = 4 * TY + 3, SW = 4 * TX + 3;        // input tile rows / cols

__device__ inline float ld(float v) { return v; }
__device__ inline float ld(__half v) { return __half2float(v); }
__device__ inline void st(float* p, float v) { *p = v; }
__device__ inline void st(__half* p, float v) { *p = __float2half_rn(v); }

template <typename T, bool HWC>
__global__ void subconv02(const T* __restrict__ x, T* __restrict__ y, const float* __restrict__ w0,
                          const float* __restrict__ b0, const float* __restrict__ w2, const float* __restrict__ b2,
                          int C, int Tn, int F, int T1, int F1, int T2, int F2, int nfx) {
  __shared__ float tile[SH][SW];
  __shared__ float sw0[CPB][9], sb0[CPB], sw2[CPB][9], sb2[CPB];
  const int b = blockIdx.z, c0 = blockIdx.y * CPB;
  const int fx = blockIdx.x % nfx, ty0 = (blockIdx.x / nfx) * TY, fx0 = fx * TX;
  const int tid = threadIdx.y * TX + threadIdx.x;
  const int r0 = 4 * ty0 - 3, q0 = 4 * fx0 - 3;
  const T* xb = x + static_cast<size_t>(b) * Tn * F;
  for (int i = tid; i < SH * SW; i += TX * TY) {
    int r = r0 + i / SW, q = q0 + i % SW;
    tile[i / SW][i % SW] = (r >= 0 && r < Tn && q >= 0 && q < F) ? ld(xb[static_cast<size_t>(r) * F + q]) : 0.f;
  }
  for (int i = tid; i < CPB * 9; i += TX * TY) {
    int c = c0 + i / 9;
    sw0[i / 9][i % 9] = c < C ? w0[c * 9 + i % 9] : 0.f;
    sw2[i / 9][i % 9] = c < C ? w2[c * 9 + i % 9] : 0.f;
  }
  for (int i = tid; i < CPB; i += TX * TY) {
    sb0[i] = c0 + i < C ? b0[c0 + i] : 0.f;
    sb2[i] = c0 + i < C ? b2[c0 + i] : 0.f;
  }
  __syncthreads();
  const int t2 = ty0 + threadIdx.y, f2 = fx0 + threadIdx.x;
  if (t2 >= T2 || f2 >= F2) return;
  float p[7][7];
#pragma unroll
  for (int i = 0; i < 7; ++i)
#pragma unroll
    for (int j = 0; j < 7; ++j) p[i][j] = tile[4 * threadIdx.y + i][4 * threadIdx.x + j];
  bool ok[3][3];                                       // conv.0 output inside its map (else conv.2's zero padding)
#pragma unroll
  for (int a = 0; a < 3; ++a)
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      int t1 = 2 * t2 - 1 + a, f1 = 2 * f2 - 1 + d;
      ok[a][d] = t1 >= 0 && t1 < T1 && f1 >= 0 && f1 < F1;
    }
  const int nc = min(CPB, C - c0);
  if (HWC) {                                           // [B, T2, F2, C(8-padded)]: 32 channels in registers
    float out[CPB];
#pragma unroll
    for (int k = 0; k < CPB; ++k) {
      float s = sb2[k];
#pragma unroll
      for (int a = 0; a < 3; ++a)
#pragma unroll
        for (int d = 0; d < 3; ++d) {
          float v = sb0[k];
#pragma unroll
          for (int u = 0; u < 3; ++u)
#pragma unroll
            for (int w = 0; w < 3; ++w) v = fmaf(sw0[k][u * 3 + w], p[2 * a + u][2 * d + w], v);
          s = fmaf(ok[a][d] ? sw2[k][a * 3 + d] : 0.f, fmaxf(v, 0.f), s);
        }
      out[k] = s;
    }
    const int Cp = (C + 7) / 8 * 8;
    T* yb = y + ((static_cast<size_t>(b) * T2 + t2) * F2 + f2) * Cp + c0;
    if (sizeof(T) == 2 && nc == CPB) {
      uint4* v = reinterpret_cast<uint4*>(yb);
#pragma unroll
      for (int q = 0; q < CPB / 8; ++q) {
        __half2 h[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) h[r] = __floats2half2_rn(out[q * 8 + 2 * r], out[q * 8 + 2 * r + 1]);
        v[q] = *reinterpret_cast<uint4*>(h);
      }
    } else {
      for (int k = 0; k < nc; ++k) st(yb + k, out[k]);
    }
    return;
  }
  const size_t plane = static_cast<size_t>(T2) * F2;
  T* yb = y + (static_cast<size_t>(b) * C + c0) * plane + static_cast<size_t>(t2) * F2 + f2;
  for (int k = 0; k < nc; ++k) {
    float s = sb2[k];
#pragma unroll
    for (int a = 0; a < 3; ++a)
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        float v = sb0[k];
#pragma unroll
        for (int u = 0; u < 3; ++u)
#pragma unroll
          for (int w = 0; w < 3; ++w) v = fmaf(sw0[k][u * 3 + w], p[2 * a + u][2 * d + w], v);
        s = fmaf(ok[a][d] ? sw2[k][a * 3 + d] : 0.f, fmaxf(v, 0.f), s);
      }
    st(yb + k * plane, s);
  }
}

// NHWC variant: one thread per output channel (blockDim = C), looping over a PT x PF tile of output positions whose
// input patch sits in shared memory, so each position's C channels are written as one coalesced C*2-byte run.
constexpr int PT = 4, PF = 8;
constexpr int NSH = 4 * PT + 3, NSW = 4 * PF + 3;

__global__ void subconv02_nhwc(const __half* __restrict__ x, __half* __restrict__ y, const float* __restrict__ w0,
                               const float* __restrict__ b0, const float* __restrict__ w2, const float* __restrict__ b2,
                               int C, int Tn, int F, int T1, int F1, int T2, int F2, int nfx,
                               const int* __restrict__ len) {
  __shared__ float tile[NSH][NSW];
  const int b = blockIdx.z, c = threadIdx.x;
  const int fx = blockIdx.x % nfx, t20 = (blockIdx.x / nfx) * PT, f20 = fx * PF;
  const int r0 = 4 * t20 - 3, q0 = 4 * f20 - 3;
  const __half* xb = x + static_cast<size_t>(b) * Tn * F;
  // masked (NeMo's MaskedConvSequential): time rows at or past each stage's length are zeroed before the next layer
  const int L0 = len ? min(len[b], Tn) : Tn, L1 = len ? (L0 + 1) / 2 : T1, L2 = len ? (L1 + 1) / 2 : T2;
  for (int i = threadIdx.x; i < NSH * NSW; i += blockDim.x) {
    int r = r0 + i / NSW, q = q0 + i % NSW;
    tile[i / NSW][i % NSW] = (r >= 0 && r < L0 && q >= 0 && q < F) ? __half2float(xb[static_cast<size_t>(r) * F + q]) : 0.f;
  }
  __syncthreads();
  if (c >= C) return;
  float k0[9], k2[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) { k0[i] = w0[c * 9 + i]; k2[i] = w2[c * 9 + i]; }
  const float bb0 = b0[c], bb2 = b2[c];
  for (int pt = 0; pt < PT; ++pt) {
    const int t2 = t20 + pt;
    if (t2 >= T2) break;
    for (int pf = 0; pf < PF; ++pf) {
      const int f2 = f20 + pf;
      if (f2 >= F2) break;
      float p[7][7];                                           // the position's input patch, read once
#pragma unroll
      for (int i = 0; i < 7; ++i)
#pragma unroll
        for (int j = 0; j < 7; ++j) p[i][j] = tile[4 * pt + i][4 * pf + j];
      float s = bb2;
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        const int t1 = 2 * t2 - 1 + a;
#pragma unroll
        for (int d = 0; d < 3; ++d) {
          const int f1 = 2 * f2 - 1 + d;
          float v = bb0;
#pragma unroll
          for (int u = 0; u < 3; ++u)
#pragma unroll
            for (int w = 0; w < 3; ++w) v = fmaf(k0[u * 3 + w], p[2 * a + u][2 * d + w], v);
          const bool in = t1 >= 0 && t1 < T1 && t1 < L1 && f1 >= 0 && f1 < F1;   // conv.2 zero padding / mask
          s = fmaf(in ? k2[a * 3 + d] : 0.f, fmaxf(v, 0.f), s);
        }
      }
      y[((static_cast<size_t>(b) * T2 + t2) * F2 + f2) * C + c] = __float2half_rn(t2 < L2 ? s : 0.f);
    }
  }
}


// conv.0 + ReLU + conv.2 + conv.3 (1x1, C -> C, bias) + ReLU, channels-last output. One block = PT3 full rows of t2
// (all F2 columns, P = PT3 * F2 <= 128 positions); 256 threads = C channels. Stage 1 is subconv02_nhwc's arithmetic
// into a shared [P][C] fp16 tile; stage 2 multiplies it by W3 on the tensor cores (WMMA 16x16x16 fp16, fp32
// accumulate), warp w owning output channels [32w, 32w + 32); the epilogue adds b3, applies ReLU and writes fp16.
// Removes conv.2's [B, T2, F2, C] write and conv.3's read of it (210 MB per 32 x 16 s batch for C = 256).
constexpr int P3 = 32;
__global__ void subconv023_nhwc(const __half* __restrict__ x, __half* __restrict__ y, const float* __restrict__ w0,
                                const float* __restrict__ b0, const float* __restrict__ w2, const float* __restrict__ b2,
                                const __half* __restrict__ w3, const float* __restrict__ b3, int C, int Tn, int F,
                                int T1, int F1, int T2, int F2, int PT3) {
  extern __shared__ __align__(128) unsigned char smem[];
  const int lda = C + 8, NSW3 = 4 * F2 + 3, NSH3 = 4 * PT3 + 3;
  __half* A = reinterpret_cast<__half*>(smem);                                   // [P3][lda]
  float* tile = reinterpret_cast<float*>(smem + static_cast<size_t>(P3) * lda * 2);   // [NSH3][NSW3]
  float* scr = tile + (NSH3 * NSW3 + 7) / 8 * 8;                                 // per warp 16 x 16 (32-byte aligned)
  const int b = blockIdx.y, t20 = blockIdx.x * PT3, c = threadIdx.x;
  const int r0 = 4 * t20 - 3;
  const __half* xb = x + static_cast<size_t>(b) * Tn * F;
  for (int i = c; i < NSH3 * NSW3; i += blockDim.x) {
    int r = r0 + i / NSW3, q = i % NSW3 - 3;
    tile[i] = (r >= 0 && r < Tn && q >= 0 && q < F) ? __half2float(xb[static_cast<size_t>(r) * F + q]) : 0.f;
  }
  __syncthreads();
  const int P = PT3 * F2, Pp = (P + 15) / 16 * 16;
  {
    float k0[9], k2[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) { k0[i] = w0[c * 9 + i]; k2[i] = w2[c * 9 + i]; }
    const float bb0 = b0[c], bb2 = b2[c];
    for (int pp = 0; pp < Pp; ++pp) {
      const int pt = pp / F2, f2 = pp % F2, t2 = t20 + pt;
      float s = 0.f;
      if (pp < P && t2 < T2) {
        s = bb2;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
          const int t1 = 2 * t2 - 1 + a;
#pragma unroll
          for (int d = 0; d < 3; ++d) {
            const int f1 = 2 * f2 - 1 + d;
            float v = bb0;
#pragma unroll
            for (int u = 0; u < 3; ++u)
#pragma unroll
              for (int w = 0; w < 3; ++w)
                v = fmaf(k0[u * 3 + w], tile[(4 * pt + 2 * a + u) * NSW3 + 4 * f2 + 2 * d + w], v);
            const bool in = t1 >= 0 && t1 < T1 && f1 >= 0 && f1 < F1;
            s = fmaf(in ? k2[a * 3 + d] : 0.f, fmaxf(v, 0.f), s);
          }
        }
      }
      A[pp * lda + c] = __float2half_rn(s);
    }
  }
  __syncthreads();
  const int warp = c / 32, lane = c % 32, MT = Pp / 16;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[P3 / 16][2];
#pragma unroll
  for (int mt = 0; mt < P3 / 16; ++mt)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[mt][j], 0.f);
  for (int k = 0; k < C; k += 16) {
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], w3 + static_cast<size_t>(warp * 32 + j * 16) * C + k, C);
#pragma unroll
    for (int mt = 0; mt < P3 / 16; ++mt) {
      if (mt < MT) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
        wmma::load_matrix_sync(af, A + mt * 16 * lda + k, lda);
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[mt][j], af, bf[j], acc[mt][j]);
      }
    }
  }
  float* sw = scr + warp * 256;
#pragma unroll
  for (int mt = 0; mt < P3 / 16; ++mt) {
    if (mt >= MT) break;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::store_matrix_sync(sw, acc[mt][j], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        const int pp = mt * 16 + e / 16, n = warp * 32 + j * 16 + e % 16;
        const int pt = pp / F2, f2 = pp % F2, t2 = t20 + pt;
        if (pp < P && t2 < T2)
          y[((static_cast<size_t>(b) * T2 + t2) * F2 + f2) * C + n] = __float2half_rn(fmaxf(sw[e] + b3[n], 0.f));
      }
      __syncwarp();
    }
  }
}

size_t subconv023_smem(int C, int F2, int PT3) {
  return static_cast<size_t>(P3) * (C + 8) * 2 + static_cast<size_t>(((4 * PT3 + 3) * (4 * F2 + 3) + 7) / 8 * 8) * 4 +
         static_cast<size_t>(C / 32) * 256 * 4;
}

}  // namespace subconv

namespace {
using namespace subconv;

struct SubWeights {
  std::vector<float> w0, b0, w2, b2;
  int C = 0;
  int32_t nhwc = 0;
  int32_t masked = 0;                                  // 1: second input is lengths int32 [B] (NeMo masking; nhwc only)
  std::vector<float> w3, b3;                           // optional conv.3 (1x1) fused in: output relu(conv3(.)) NHWC
  float* d = nullptr;
  __half* d3 = nullptr;                                // w3 fp16 [C][C]
  float* db3 = nullptr;                                  // device copy: w0 | b0 | w2 | b2
  ~SubWeights() {
    if (d) cudaFree(d);
    if (d3) cudaFree(d3);
    if (db3) cudaFree(db3);
  }
  bool upload() {
    if (d) return true;
    size_t n = 20 * static_cast<size_t>(C);
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return false;
    std::vector<float> h;
    h.reserve(n);
    for (auto* v : {&w0, &b0, &w2, &b2}) h.insert(h.end(), v->begin(), v->end());
    if (cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (!w3.empty()) {
      std::vector<__half> h3(w3.size());
      for (size_t i = 0; i < w3.size(); ++i) h3[i] = __float2half_rn(w3[i]);
      if (cudaMalloc(&d3, h3.size() * 2) != cudaSuccess || cudaMalloc(&db3, b3.size() * 4) != cudaSuccess ||
          cudaMemcpy(d3, h3.data(), h3.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(db3, b3.data(), b3.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    }
    return true;
  }
};

int out_len(int n) { return (n - 1) / 2 + 1; }

class SubConv02 : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  explicit SubConv02(std::shared_ptr<SubWeights> w) : w_(std::move(w)) {
    fields_ = {PluginField("w0", w_->w0.data(), PluginFieldType::kFLOAT32, 9 * w_->C),
               PluginField("b0", w_->b0.data(), PluginFieldType::kFLOAT32, w_->C),
               PluginField("w2", w_->w2.data(), PluginFieldType::kFLOAT32, 9 * w_->C),
               PluginField("b2", w_->b2.data(), PluginFieldType::kFLOAT32, w_->C),
               PluginField("nhwc", &w_->nhwc, PluginFieldType::kINT32, 1),
               PluginField("masked", &w_->masked, PluginFieldType::kINT32, 1),
               PluginField("w3", w_->w3.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(w_->w3.size())),
               PluginField("b3", w_->b3.data(), PluginFieldType::kFLOAT32, static_cast<int32_t>(w_->b3.size()))};
    fc_.nbFields = w_->w3.empty() ? 6 : 8;
    fc_.fields = fields_.data();
  }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
    if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
    if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
  }
  IPluginV3* clone() noexcept override { return new SubConv02(w_); }
  AsciiChar const* getPluginName() const noexcept override { return "SubConv02"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) noexcept override { return w_->upload() ? 0 : -1; }
  int32_t getOutputDataTypes(DataType* out, int32_t, DataType const* in, int32_t) const noexcept override {
    out[0] = in[0];
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t,
                          IExprBuilder& eb) noexcept override {
    auto half = [&](IDimensionExpr const* n) {
      auto one = eb.constant(1), two = eb.constant(2);
      return eb.operation(DimensionOperation::kSUM,
                          *eb.operation(DimensionOperation::kFLOOR_DIV,
                                        *eb.operation(DimensionOperation::kSUB, *n, *one), *two), *one);
    };
    out[0].nbDims = 4;
    out[0].d[0] = in[0].d[0];
    if (w_->nhwc) {
      out[0].d[1] = half(half(in[0].d[2]));
      out[0].d[2] = half(half(in[0].d[3]));
      out[0].d[3] = eb.constant(w_->C);
    } else {
      out[0].d[1] = eb.constant(w_->C);
      out[0].d[2] = half(half(in[0].d[2]));
      out[0].d[3] = half(half(in[0].d[3]));
    }
    return 0;
  }
  // fp16 only (fp32 doubles the output traffic); output linear NCHW or channels-last HWC8 (what TensorRT's
  // following 1x1 conv wants, which saves a layout conversion)
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t nbIn, int32_t) noexcept override {
    if (w_->masked && pos == 1) return io[1].desc.type == DataType::kINT32 && io[1].desc.format == TensorFormat::kLINEAR;
    if (io[pos].desc.type != DataType::kHALF) return false;
    if (pos == 0) return io[0].desc.format == TensorFormat::kLINEAR;
    if (w_->nhwc) return io[pos].desc.format == TensorFormat::kLINEAR;
    return io[pos].desc.format == TensorFormat::kLINEAR || io[pos].desc.format == TensorFormat::kHWC8;
  }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    return w_->upload() ? 0 : -1;
  }
  int32_t enqueue(PluginTensorDesc const* id, PluginTensorDesc const* od, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (!w_->upload()) return -1;
    const int B = id[0].dims.d[0], Tn = id[0].dims.d[2], F = id[0].dims.d[3], C = w_->C;
    const int T1 = out_len(Tn), F1 = out_len(F), T2 = out_len(T1), F2 = out_len(F1);
    if (B == 0 || Tn == 0) return 0;
    const int nfx = (F2 + TX - 1) / TX;
    dim3 grid(nfx * ((T2 + TY - 1) / TY), (C + CPB - 1) / CPB, B), block(TX, TY);
    const float* d = w_->d;
    auto x = static_cast<const __half*>(inputs[0]);
    auto yy = static_cast<__half*>(outputs[0]);
    if (!w_->w3.empty()) {                                  // conv.0..conv.3 + ReLU fused (subconv023_nhwc)
      const int PT3 = std::max(1, P3 / F2);
      const size_t sm = subconv023_smem(C, F2, PT3);
      static bool attr = false;
      if (!attr) {
        cudaFuncSetAttribute(subconv023_nhwc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sm));
        attr = true;
      }
      dim3 g3((T2 + PT3 - 1) / PT3, B);
      subconv023_nhwc<<<g3, C, sm, stream>>>(x, yy, d, d + 9 * C, d + 10 * C, d + 19 * C, w_->d3, w_->db3, C, Tn, F,
                                             T1, F1, T2, F2, PT3);
    } else if (w_->nhwc) {
      const int nfx2 = (F2 + PF - 1) / PF;
      dim3 g2(nfx2 * ((T2 + PT - 1) / PT), 1, B);
      subconv02_nhwc<<<g2, C, 0, stream>>>(x, yy, d, d + 9 * C, d + 10 * C, d + 19 * C, C, Tn, F, T1, F1, T2, F2, nfx2,
                                           w_->masked ? static_cast<const int*>(inputs[1]) : nullptr);
    } else if (od[0].format == TensorFormat::kHWC8)
      subconv02<__half, true><<<grid, block, 0, stream>>>(x, yy, d, d + 9 * C, d + 10 * C, d + 19 * C, C, Tn, F, T1, F1, T2, F2, nfx);
    else
      subconv02<__half, false><<<grid, block, 0, stream>>>(x, yy, d, d + 9 * C, d + 10 * C, d + 19 * C, C, Tn, F, T1, F1, T2, F2, nfx);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "SubConv02: %s\n", cudaGetErrorString(e)); return -2; }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fc_; }

 private:
  std::shared_ptr<SubWeights> w_;
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
};

class SubConv02Creator : public IPluginCreatorV3One {
 public:
  SubConv02Creator() {
    attrs_ = {PluginField("w0", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("b0", nullptr, PluginFieldType::kFLOAT32, 0),
              PluginField("w2", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("b2", nullptr, PluginFieldType::kFLOAT32, 0),
              PluginField("nhwc", nullptr, PluginFieldType::kINT32, 1), PluginField("masked", nullptr, PluginFieldType::kINT32, 1),
              PluginField("w3", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("b3", nullptr, PluginFieldType::kFLOAT32, 0)};
    fc_.nbFields = 8;
    fc_.fields = attrs_.data();
  }
  IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    auto w = std::make_shared<SubWeights>();
    int have = 0;
    for (int i = 0; i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      std::string n = f.name;
      if (n == "masked") {
        w->masked = f.type == PluginFieldType::kINT64 ? static_cast<int32_t>(*static_cast<int64_t const*>(f.data))
                                                      : *static_cast<int32_t const*>(f.data);
        continue;
      }
      if (n != "nhwc" && f.type != PluginFieldType::kFLOAT32) {
        fprintf(stderr, "SubConv02 %s: field %s must be float32\n", name, f.name);
        return nullptr;
      }
      auto p = static_cast<float const*>(f.data);
      if (n == "nhwc") {
        w->nhwc = f.type == PluginFieldType::kINT64 ? static_cast<int32_t>(*static_cast<int64_t const*>(f.data))
                                                    : *static_cast<int32_t const*>(f.data);
        continue;
      }
      if (n == "w3" || n == "b3") {
        (n == "w3" ? w->w3 : w->b3).assign(p, p + f.length);
        continue;
      }
      std::vector<float>* dst = n == "w0" ? &w->w0 : n == "b0" ? &w->b0 : n == "w2" ? &w->w2 : n == "b2" ? &w->b2 : nullptr;
      if (!dst) continue;
      dst->assign(p, p + f.length);
      ++have;
    }
    w->C = static_cast<int>(w->b0.size());
    if (have != 4 || w->w0.size() != 9u * w->C || w->w2.size() != 9u * w->C || w->b2.size() != static_cast<size_t>(w->C)) {
      fprintf(stderr, "SubConv02 %s: bad or missing fields\n", name);
      return nullptr;
    }
    if (!w->w3.empty() && (!w->nhwc || w->masked || w->C % 32 || w->C > 1024 ||
                           w->w3.size() != static_cast<size_t>(w->C) * w->C || w->b3.size() != static_cast<size_t>(w->C))) {
      fprintf(stderr, "SubConv02 %s: fused conv.3 needs nhwc, no mask, C %% 32 == 0 and w3 [C*C], b3 [C]\n", name);
      return nullptr;
    }
    if (w->masked && !w->nhwc) {
      fprintf(stderr, "SubConv02 %s: masked needs nhwc\n", name);
      return nullptr;
    }
    return new SubConv02(w);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  AsciiChar const* getPluginName() const noexcept override { return "SubConv02"; }
  AsciiChar const* getPluginVersion() const noexcept override { return "1"; }
  AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

 private:
  std::vector<PluginField> attrs_;
  PluginFieldCollection fc_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(SubConv02Creator);
