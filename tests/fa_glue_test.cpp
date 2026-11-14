// Engine-level check of VulkanBackend::q35_linear_layer (whole Qwen3.5 linear-attention + FFN layer, one submit, state resident on the card)
// against a double-precision reference built from the dequantised weights.
#include "../src/gpu/vulkan_backend.h"
#include "../src/model/kvcache.h"
#include <random>
#include <memory>
#include <cmath>
using namespace nexus;
using QT = quant::QuantType;
static std::vector<std::unique_ptr<std::vector<uint8_t>>> keep;
static std::mt19937 rng(11);
static const void* gen(QT qt, size_t rows, size_t cols, std::vector<float>& deq) {
  size_t bs, bw; switch (qt) { case QT::Q8_0: bs=34; bw=32; break; case QT::Q4_K: bs=144; bw=256; break; case QT::Q5_K: bs=176; bw=256; break; default: bs=210; bw=256; }
  size_t nb = cols / bw; keep.emplace_back(new std::vector<uint8_t>(rows*nb*bs)); auto& W = *keep.back(); std::uniform_int_distribution<int> bi(0,255);
  for (auto& v : W) v = (uint8_t)bi(rng);
  for (size_t i = 0; i < rows*nb; ++i) { uint16_t d = 0x1400 + (bi(rng)&0x3f), dm = 0x1400 + (bi(rng)&0x3f); size_t o = i*bs;
    if (qt==QT::Q8_0) memcpy(&W[o],&d,2); else if (qt==QT::Q6_K) memcpy(&W[o+208],&d,2); else { memcpy(&W[o],&d,2); memcpy(&W[o+2],&dm,2); } }
  deq.resize(rows*cols); for (size_t r = 0; r < rows; ++r) quant::dequantize_row(W.data(), &deq[r*cols], r, cols, qt);
  return W.data();
}
static void mv(const std::vector<float>& W, size_t rows, size_t cols, const std::vector<double>& x, std::vector<double>& y) { y.assign(rows,0); for (size_t r=0;r<rows;r++){ double a=0; for(size_t c=0;c<cols;c++) a+=(double)W[r*cols+c]*x[c]; y[r]=a; } }
static std::vector<double> rms(const std::vector<double>& x, const std::vector<float>& w, double eps){ double ss=0; for(double v:x) ss+=v*v; double inv=1.0/std::sqrt(ss/x.size()+eps); std::vector<double> y(x.size()); for(size_t i=0;i<x.size();i++) y[i]=x[i]*inv*w[i]; return y; }
static double sg(double x){ return 1.0/(1.0+std::exp(-x)); }
static double relerr(const std::vector<float>& a, const std::vector<double>& b){ double se=0,sr=0; for(size_t i=0;i<a.size();i++){ se+=(a[i]-b[i])*(a[i]-b[i]); sr+=b[i]*b[i]; } return std::sqrt(se/std::max(sr,1e-30)); }
static uint16_t F16(float f) {   // same (truncating) conversion as KVCache::f32_to_f16
  uint32_t x; memcpy(&x, &f, 4); uint32_t sign = (x >> 31) & 1; int32_t expo = (int32_t)((x >> 23) & 0xFF) - 127 + 15; uint32_t mant = (x >> 13) & 0x3FF;
  if (expo <= 0) return (uint16_t)(sign << 15); if (expo >= 31) return (uint16_t)((sign << 15) | 0x7C00); return (uint16_t)((sign << 15) | ((uint32_t)expo << 10) | mant); }
static float    F32(uint16_t h) { return quant::f16_to_f32(h); }

struct RefLayer {
  uint32_t hidden=512, nh=8, nkv=2, hd=256, rot=64, ffn=1024; double eps=1e-6;
  std::vector<float> dq, dk, dv, dwo, dg, du, dd, an, fn, qn, kn;
  bool qzc=true, kzc=false;
};


int main() {
  auto& g = gpu::gpu(); if (!g.init(false)) { printf("no gpu\n"); return 2; }
  RefLayer R; const uint32_t hidden=R.hidden, nh=R.nh, nkv=R.nkv, hd=R.hd, rot=R.rot, ffn=R.ffn; const double eps=R.eps;
  const uint32_t q_dim=nh*hd, kv_dim=nkv*hd, half=rot/2, G=nh/nkv;
  g.init_activation_bufs(hidden, q_dim, kv_dim, ffn);
  gpu::VulkanBackend::Q35FullDesc d; d.owner=&g; d.layer=2; d.hidden=hidden; d.n_heads=nh; d.n_kv=nkv; d.hd=hd; d.rot=rot; d.ffn=ffn; d.norm_eps=(float)eps; d.scale=0;
  d.wq=gen(QT::Q5_K,2*q_dim,hidden,R.dq); d.wq_t=QT::Q5_K; d.wk=gen(QT::Q6_K,kv_dim,hidden,R.dk); d.wk_t=QT::Q6_K; d.wv=gen(QT::Q8_0,kv_dim,hidden,R.dv); d.wv_t=QT::Q8_0;
  d.wo=gen(QT::Q4_K,hidden,q_dim,R.dwo); d.wo_t=QT::Q4_K; d.gate=gen(QT::Q5_K,ffn,hidden,R.dg); d.gate_t=QT::Q5_K; d.up=gen(QT::Q5_K,ffn,hidden,R.du); d.up_t=QT::Q5_K;
  d.down=gen(QT::Q4_K,hidden,ffn,R.dd); d.down_t=QT::Q4_K;
  std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ud(0,1);
  R.an.resize(hidden); R.fn.resize(hidden); R.qn.resize(hd); R.kn.resize(hd);
  for (auto& v:R.an) v=0.5f+ud(rng); for (auto& v:R.fn) v=0.5f+ud(rng); for (auto& v:R.qn) v=nd(rng)*0.3f; for (auto& v:R.kn) v=0.5f+ud(rng);
  d.attn_norm=R.an.data(); d.ffn_norm=R.fn.data(); d.q_norm=R.qn.data(); d.k_norm=R.kn.data(); d.q_zc=R.qzc; d.k_zc=R.kzc;

  model::KVCache kc; kc.init(3, nkv, hd);       // layer 2 is the one under test
  const int L = 2; int bad = 0;
  std::vector<float> kk(kv_dim), vv(kv_dim);
  auto host_store = [&](int pos) { for (auto& v:kk) v=nd(rng)*1.5f; for (auto& v:vv) v=nd(rng); kc.store(L, pos, kk.data(), vv.data(), kv_dim); };
  // one engine-style decode step at `pos` (mirrors Transformer::q35_gpu_full_step) + double-precision check against the host cache
  auto step = [&](int pos, const char* what) {
    std::vector<float> tab(rot);
    for (uint32_t i=0;i<half;i++){ double fr=1.0/std::pow(1.0e6,(double)(2*i)/(double)rot); double th=(double)pos*fr; tab[i]=(float)std::cos(th); tab[half+i]=(float)std::sin(th); }
    d.rope_tab=tab.data(); d.pos=pos;
    const uint16_t* kp[2]; const uint16_t* vp[2]; for (uint32_t h=0;h<nkv;h++){ kp[h]=kc.k_ptr(L,h); vp[h]=kc.v_ptr(L,h); } d.k_host=kp; d.v_host=vp;
    d.host_tokens = kc.n_tokens(); d.host_synced = (uint32_t)std::min<size_t>(kc.mirror_valid(L), (size_t)pos);
    std::vector<double> xr(hidden); std::vector<float> x(hidden); for (uint32_t i=0;i<hidden;i++){ xr[i]=nd(rng); x[i]=(float)xr[i]; }
    std::vector<float> kvo(2*kv_dim); uint32_t mvl=0; const uint32_t synced_in = d.host_synced;
    bool ok = g.q35_full_layer(d, x.data(), kvo.data(), &mvl);
    if (!ok) { printf("%-46s pos %5d  backend refused  FAIL\n", what, pos); bad++; return; }
    kc.store(L, pos, kvo.data(), kvo.data()+kv_dim, kv_dim); kc.set_mirror_valid(L, mvl);
    // reference with the host cache (now including this token)
    std::vector<double> xb = rms(xr,R.an,eps), qg; mv(R.dq,2*q_dim,hidden,xb,qg);
    std::vector<double> q(q_dim), gate(q_dim);
    for (uint32_t h=0; h<nh; ++h) { double ss=0; for(uint32_t i=0;i<hd;i++) ss+=qg[h*2*hd+i]*qg[h*2*hd+i]; double inv=1.0/std::sqrt(ss/hd+eps);
      for(uint32_t i=0;i<hd;i++) q[h*hd+i]=qg[h*2*hd+i]*inv*(1.0+R.qn[i]); for (uint32_t i=0;i<half;i++){ double c=tab[i], s=tab[half+i]; double a=q[h*hd+i], b=q[h*hd+i+half]; q[h*hd+i]=a*c-b*s; q[h*hd+i+half]=b*c+a*s; }
      for(uint32_t i=0;i<hd;i++) gate[h*hd+i]=qg[h*2*hd+hd+i]; }
    std::vector<double> att(q_dim); std::vector<float> row(hd);
    for (uint32_t h=0; h<nh; ++h) { const uint32_t kvh=h/G; std::vector<double> s(pos+1); double mx=-1e300;
      for (int t=0;t<=pos;t++){ kc.get_k_f32(L,t,kvh,row.data()); double a=0; for(uint32_t i=0;i<hd;i++) a+=q[h*hd+i]*row[i]; s[t]=a/std::sqrt((double)hd); mx=std::max(mx,s[t]); }
      double l=0; for(auto& v:s){ v=std::exp(v-mx); l+=v; }
      for (uint32_t i=0;i<hd;i++){ double a=0; for(int t=0;t<=pos;t++){ kc.get_v_f32(L,t,kvh,row.data()); a+=s[t]*row[i]; } att[h*hd+i]=a/l*sg(gate[h*hd+i]); } }
    std::vector<double> o; mv(R.dwo,hidden,q_dim,att,o); for(uint32_t i=0;i<hidden;i++) xr[i]+=o[i];
    std::vector<double> xn=rms(xr,R.fn,eps), gt,up,hm(ffn),dn; mv(R.dg,ffn,hidden,xn,gt); mv(R.du,ffn,hidden,xn,up); for(uint32_t i=0;i<ffn;i++) hm[i]=gt[i]/(1.0+std::exp(-gt[i]))*up[i];
    mv(R.dd,hidden,ffn,hm,dn); for(uint32_t i=0;i<hidden;i++) xr[i]+=dn[i];
    double e = relerr(x, xr); bool pass = e < 3e-3; if (!pass) bad++;
    printf("%-46s pos %5d  synced_in %5u  x rel err %.2e  %s\n", what, pos, synced_in, e, pass?"PASS":"FAIL");
  };
  int pos = 0;
  for (; pos < 700; ++pos) host_store(pos);                       // "prefill": host-only
  for (int i = 0; i < 3; ++i, ++pos) step(pos, "decode after host prefill");
  for (int i = 0; i < 2; ++i, ++pos) { host_store(pos); }          // more host-only tokens (e.g. CPU or batched path ran a few tokens)
  step(pos, "decode after 2 more host-only tokens"); ++pos;
  { int back = pos - 40; pos = back; for (int i = 0; i < 2; ++i) { step(pos, "rewind 40 tokens, decode again"); ++pos; } }
  { for (int i = 0; i < 40; ++i) { std::vector<float> f(hd); for (auto& v:f) v=nd(rng)*3; kc.set_k_f32(L, 100, 1, f.data()); } step(pos, "old K edited on host (rope rescale style)"); ++pos; }
  { kc.clear(); pos = 0; for (; pos < 50; ++pos) host_store(pos); step(pos, "after clear + new 50-token prompt"); ++pos; }
  printf(bad ? "FAILED %d\n" : "all passed\n", bad); return bad?1:0;
}
