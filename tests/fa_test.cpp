// Engine-level check of VulkanBackend::q35_linear_layer (whole Qwen3.5 linear-attention + FFN layer, one submit, state resident on the card)
// against a double-precision reference built from the dequantised weights.
#include "../src/gpu/vulkan_backend.h"
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
  const uint32_t q_dim=nh*hd, kv_dim=nkv*hd;
  g.init_activation_bufs(hidden, q_dim, kv_dim, ffn);
  gpu::VulkanBackend::Q35FullDesc d; d.owner=&g; d.hidden=hidden; d.n_heads=nh; d.n_kv=nkv; d.hd=hd; d.rot=rot; d.ffn=ffn; d.norm_eps=(float)eps; d.scale=0;
  d.wq=gen(QT::Q5_K,2*q_dim,hidden,R.dq); d.wq_t=QT::Q5_K; d.wk=gen(QT::Q6_K,kv_dim,hidden,R.dk); d.wk_t=QT::Q6_K; d.wv=gen(QT::Q8_0,kv_dim,hidden,R.dv); d.wv_t=QT::Q8_0;
  d.wo=gen(QT::Q4_K,hidden,q_dim,R.dwo); d.wo_t=QT::Q4_K; d.gate=gen(QT::Q5_K,ffn,hidden,R.dg); d.gate_t=QT::Q5_K; d.up=gen(QT::Q5_K,ffn,hidden,R.du); d.up_t=QT::Q5_K;
  d.down=gen(QT::Q4_K,hidden,ffn,R.dd); d.down_t=QT::Q4_K;
  std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ud(0,1);
  R.an.resize(hidden); R.fn.resize(hidden); R.qn.resize(hd); R.kn.resize(hd);
  for (auto& v:R.an) v=0.5f+ud(rng); for (auto& v:R.fn) v=0.5f+ud(rng); for (auto& v:R.qn) v=nd(rng)*0.3f; for (auto& v:R.kn) v=0.5f+ud(rng);
  d.attn_norm=R.an.data(); d.ffn_norm=R.fn.data(); d.q_norm=R.qn.data(); d.k_norm=R.kn.data(); d.q_zc=R.qzc; d.k_zc=R.kzc;
  for (auto& w : R.dq) w *= 1.0f; // (dequantised values as generated)

  int bad = 0;
  struct Scn { int layer; uint32_t prefill; int steps; const char* name; };
  const Scn scns[] = { {0, 0, 3, "first tokens (pos 0..2)"}, {1, 1500, 6, "1500-token host prefill (12 chunks, partial last), then decode"},
                       {2, 2044, 8, "crosses the 2048-token initial capacity (grow + refill from host)"}, {3, 20000, 2, "20000 tokens (157 chunks)"} };
  for (const Scn& sc : scns) {
    d.layer = sc.layer;
    // host cache, per head, f16 [tokens][hd] with generous capacity
    const size_t capT = sc.prefill + sc.steps + 8;
    std::vector<std::vector<uint16_t>> hk(nkv, std::vector<uint16_t>(capT*hd)), hv(nkv, std::vector<uint16_t>(capT*hd));
    for (uint32_t h=0; h<nkv; ++h) for (size_t i=0; i<(size_t)sc.prefill*hd; ++i) { hk[h][i]=F16(nd(rng)*1.5f); hv[h][i]=F16(nd(rng)); }
    const uint16_t* kp[2] = { hk[0].data(), hk[1].data() }; const uint16_t* vp[2] = { hv[0].data(), hv[1].data() };
    d.k_host = kp; d.v_host = vp;
    uint32_t synced = 0;
    for (int st = 0; st < sc.steps; ++st) {
      const uint32_t pos = sc.prefill + st;
      // rope table for this position (host side, same as the engine does)
      std::vector<float> tab(rot); const uint32_t half = rot/2;
      for (uint32_t i=0;i<half;i++){ double fr = 1.0/std::pow(1.0e6, (double)(2*i)/(double)rot); double th=(double)pos*fr; tab[i]=(float)std::cos(th); tab[half+i]=(float)std::sin(th); }
      d.rope_tab = tab.data(); d.pos = pos; d.host_tokens = pos; d.host_synced = synced;
      std::vector<double> xr(hidden); std::vector<float> x(hidden); for (uint32_t i=0;i<hidden;i++){ xr[i]=nd(rng); x[i]=(float)xr[i]; }
      // ---- reference (double) ----
      std::vector<double> xb = rms(xr,R.an,eps), qg,kk,vv; mv(R.dq,2*q_dim,hidden,xb,qg); mv(R.dk,kv_dim,hidden,xb,kk); mv(R.dv,kv_dim,hidden,xb,vv);
      std::vector<double> q(q_dim), gate(q_dim);
      auto headnorm = [&](const double* in, const std::vector<float>& w, bool zc, double* out) { double ss=0; for(uint32_t i=0;i<hd;i++) ss+=in[i]*in[i]; double inv=1.0/std::sqrt(ss/hd+eps); for(uint32_t i=0;i<hd;i++) out[i]=in[i]*inv*(zc? 1.0+w[i] : w[i]); };
      auto rope = [&](double* x0) { for (uint32_t i=0;i<half;i++){ double c=tab[i], s=tab[half+i]; double a=x0[i], b=x0[i+half]; x0[i]=a*c-b*s; x0[i+half]=b*c+a*s; } };
      for (uint32_t h=0; h<nh; ++h) { headnorm(&qg[(size_t)h*2*hd], R.qn, R.qzc, &q[(size_t)h*hd]); rope(&q[(size_t)h*hd]); for(uint32_t i=0;i<hd;i++) gate[(size_t)h*hd+i]=qg[(size_t)h*2*hd+hd+i]; }
      std::vector<double> kr(kv_dim); for (uint32_t h=0; h<nkv; ++h) { headnorm(&kk[(size_t)h*hd], R.kn, R.kzc, &kr[(size_t)h*hd]); rope(&kr[(size_t)h*hd]); }
      // store (f16-rounded) in the host cache
      std::vector<float> kref_f(kv_dim), vref_f(kv_dim);
      for (uint32_t h=0; h<nkv; ++h) for (uint32_t i=0;i<hd;i++) { kref_f[h*hd+i]=(float)kr[h*hd+i]; vref_f[h*hd+i]=(float)vv[h*hd+i]; }
      std::vector<std::vector<uint16_t>> hk2 = hk, hv2 = hv;   // reference copies including the new token
      for (uint32_t h=0; h<nkv; ++h) for (uint32_t i=0;i<hd;i++) { hk2[h][(size_t)pos*hd+i]=F16(kref_f[h*hd+i]); hv2[h][(size_t)pos*hd+i]=F16(vref_f[h*hd+i]); }
      std::vector<double> att(q_dim);
      const double scale = 1.0/std::sqrt((double)hd); const uint32_t G = nh/nkv;
      for (uint32_t h=0; h<nh; ++h) { const uint32_t kvh=h/G; std::vector<double> s(pos+1); double mx=-1e300;
        for (uint32_t t=0;t<=pos;t++){ double a=0; for(uint32_t i=0;i<hd;i++) a+=q[(size_t)h*hd+i]*F32(hk2[kvh][(size_t)t*hd+i]); s[t]=a*scale; mx=std::max(mx,s[t]); }
        double l=0; for (auto& v:s){ v=std::exp(v-mx); l+=v; }
        for (uint32_t i=0;i<hd;i++){ double a=0; for(uint32_t t=0;t<=pos;t++) a+=s[t]*F32(hv2[kvh][(size_t)t*hd+i]); att[(size_t)h*hd+i]=a/l*sg(gate[(size_t)h*hd+i]); } }
      std::vector<double> o; mv(R.dwo,hidden,q_dim,att,o); for(uint32_t i=0;i<hidden;i++) xr[i]+=o[i];
      std::vector<double> xn=rms(xr,R.fn,eps), gt,up,hm(ffn),dn; mv(R.dg,ffn,hidden,xn,gt); mv(R.du,ffn,hidden,xn,up); for(uint32_t i=0;i<ffn;i++) hm[i]=gt[i]/(1.0+std::exp(-gt[i]))*up[i];
      mv(R.dd,hidden,ffn,hm,dn); for(uint32_t i=0;i<hidden;i++) xr[i]+=dn[i];
      // ---- backend ----
      std::vector<float> kvo(2*kv_dim); uint32_t mv_out = 0;
      bool ok = g.q35_full_layer(d, x.data(), kvo.data(), &mv_out);
      double e = ok ? relerr(x, xr) : 1.0;
      double ek=0, ekr=0, ev=0, evr=0; if (ok) for (uint32_t i=0;i<kv_dim;i++){ ek+=(kvo[i]-kref_f[i])*(kvo[i]-kref_f[i]); ekr+=kref_f[i]*kref_f[i]; ev+=(kvo[kv_dim+i]-vref_f[i])*(kvo[kv_dim+i]-vref_f[i]); evr+=vref_f[i]*vref_f[i]; }
      bool pass = ok && e < 3e-3 && std::sqrt(ek/ekr) < 3e-3 && std::sqrt(ev/evr) < 3e-3 && mv_out == pos+1;
      if (!pass) bad++;
      printf("[%s] pos %u  ok=%d  x rel err %.3e  k %.2e v %.2e  mirror_valid %u  %s\n", sc.name, pos, ok, e, ok?std::sqrt(ek/ekr):1.0, ok?std::sqrt(ev/evr):1.0, mv_out, pass?"PASS":"FAIL");
      if (!ok) break;
      // the engine stores the returned K/V on the host and tells the mirror it is in sync
      for (uint32_t h=0; h<nkv; ++h) for (uint32_t i=0;i<hd;i++) { hk[h][(size_t)pos*hd+i]=F16(kvo[h*hd+i]); hv[h][(size_t)pos*hd+i]=F16(kvo[kv_dim+h*hd+i]); }
      synced = mv_out;
    }
    printf("   mirror used %.1f MB\n", g.kv_mirror_used()/1048576.0);
  }
  // rewind: re-run position 1500+2 on layer 1 with a different token after the host overwrote it (host_synced lowered)
  // budget: tiny budget must refuse cleanly without touching x
  { g.set_gpu_kv_mb(1); auto bd = d; bd.layer = 9; bd.pos = 5000; bd.host_tokens = 5000; bd.host_synced = 0; std::vector<float> x(hidden, 0.5f), x0(x), kvo(2*kv_dim); uint32_t mvo=0;
    std::vector<float> tab(rot, 0.f); bd.rope_tab = tab.data();
    std::vector<uint16_t> z((size_t)5000*hd, 0); const uint16_t* zp[2] = { z.data(), z.data() }; bd.k_host = zp; bd.v_host = zp;
    bool ok = g.q35_full_layer(bd, x.data(), kvo.data(), &mvo); printf("1 MB budget refuses 5000 tokens: %s\n", (!ok && x==x0) ? "yes" : "NO"); if (ok || x!=x0) bad++;
    g.set_gpu_kv_mb(0); ok = g.q35_full_layer(d, x.data(), kvo.data(), &mvo); printf("mode off refuses: %s\n", !ok ? "yes" : "NO"); if (ok) bad++; }
  printf(bad ? "FAILED %d\n" : "all passed\n", bad); return bad?1:0;
}
