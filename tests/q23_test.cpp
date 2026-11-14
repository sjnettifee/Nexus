// Engine-level check of Q2_K / Q3_K: GPU backend (mat_vec, mat_mat) and CPU path versus a reference built from the gguf-py verified dequantiser.
#include "../src/gpu/vulkan_backend.h"
#include "../src/cpu/matmul.h"
#include <random>
#include <memory>
using namespace nexus;
using QT = quant::QuantType;
static double relerr(const std::vector<float>& a, const std::vector<float>& b){ double se=0,sr=0; for(size_t i=0;i<a.size();i++){ se+=(double)(a[i]-b[i])*(a[i]-b[i]); sr+=(double)b[i]*b[i]; } return std::sqrt(se/std::max(sr,1e-30)); }
static std::vector<std::unique_ptr<std::vector<uint8_t>>> keep;   // the backend caches weights by address: never reuse one
int main(){
  auto& g = gpu::gpu(); if(!g.init(false)) { printf("no gpu\n"); return 2; }
  std::mt19937 rng(5); int bad = 0;
  for (QT qt : {QT::Q2_K, QT::Q3_K}) {
    const size_t bs = qt==QT::Q2_K ? 84 : 110;
    for (size_t rows : {37u, 300u}) for (size_t cols : {256u, 1536u}) for (size_t nt : {1u, 5u, 140u}) {
      size_t nb = cols/256; keep.emplace_back(new std::vector<uint8_t>(rows*nb*bs)); std::vector<uint8_t>& W = *keep.back(); std::uniform_int_distribution<int> bi(0,255);
      for (auto& v : W) v = (uint8_t)bi(rng);
      for (size_t i = 0; i < rows*nb; ++i) { uint16_t d = 0x1400 + (bi(rng)&0xff), dm = 0x1400 + (bi(rng)&0xff); size_t off = i*bs; if (qt==QT::Q2_K) { memcpy(&W[off+80],&d,2); memcpy(&W[off+82],&dm,2);} else memcpy(&W[off+108],&d,2); }
      std::vector<float> X(cols*nt); std::normal_distribution<float> nd(0,1); for (auto& v : X) v = nd(rng);
      // reference: dequantise each row with quant.h (verified against gguf-py), double dot
      std::vector<float> ref(rows*nt), Wf(rows*cols);
      for (size_t r = 0; r < rows; ++r) quant::dequantize_row(W.data(), &Wf[r*cols], r, cols, qt);
      for (size_t t = 0; t < nt; ++t) for (size_t r = 0; r < rows; ++r) { double a = 0; for (size_t c = 0; c < cols; ++c) a += (double)Wf[r*cols+c]*X[t*cols+c]; ref[t*rows+r] = (float)a; }
      std::vector<float> Y(rows*nt), Yc(rows*nt);
      bool okg = (nt==1) ? g.mat_vec(W.data(), X.data(), Y.data(), rows, cols, qt) : g.mat_mat(W.data(), X.data(), Y.data(), rows, cols, nt, qt);
      if (nt==1) cpu::mat_vec(W.data(), X.data(), Yc.data(), rows, cols, qt, 4); else cpu::mat_mat(W.data(), X.data(), Yc.data(), rows, cols, nt, qt, 4);
      double eg = okg ? relerr(Y, ref) : 1.0, ec = relerr(Yc, ref);
      bool ok = okg && eg < 1e-5 && ec < 1e-5; if (!ok) bad++;
      printf("%s %4zux%-5zu tok%-4zu  gpu %s rel err %.2e   cpu rel err %.2e  %s\n", qt==QT::Q2_K?"Q2_K":"Q3_K", rows, cols, nt, okg?"ok ":"REJECTED", eg, ec, ok?"PASS":"FAIL");
    }
  }
  printf(bad ? "FAILED %d\n" : "all passed\n", bad); return bad ? 1 : 0;
}
