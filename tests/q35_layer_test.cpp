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

int main() {
  auto& g = gpu::gpu(); if (!g.init(false)) { printf("no gpu\n"); return 2; }
  const uint32_t hidden=512, kh=2, vh=4, hd=128, key_dim=kh*hd, value_dim=vh*hd, conv_dim=2*key_dim+value_dim, kernel=4, ffn=1024; const double eps=1e-6;
  g.init_activation_bufs(hidden, 2048, 512, ffn);
  std::vector<float> dq,dz,da,db,dout,dg,du,dd;
  gpu::VulkanBackend::Q35LinearDesc d; d.owner=&g; d.layer=3; d.hidden=hidden; d.conv_dim=conv_dim; d.key_dim=key_dim; d.value_dim=value_dim; d.kh=kh; d.vh=vh; d.hd=hd; d.kernel=kernel; d.ffn=ffn; d.norm_eps=(float)eps;
  d.qkv=gen(QT::Q5_K,conv_dim,hidden,dq); d.qkv_t=QT::Q5_K; d.z=gen(QT::Q6_K,value_dim,hidden,dz); d.z_t=QT::Q6_K; d.a=gen(QT::Q8_0,vh,hidden,da); d.a_t=QT::Q8_0; d.b=gen(QT::Q8_0,vh,hidden,db); d.b_t=QT::Q8_0;
  d.out=gen(QT::Q4_K,hidden,value_dim,dout); d.out_t=QT::Q4_K; d.gate=gen(QT::Q5_K,ffn,hidden,dg); d.gate_t=QT::Q5_K; d.up=gen(QT::Q5_K,ffn,hidden,du); d.up_t=QT::Q5_K; d.down=gen(QT::Q6_K,hidden,ffn,dd); d.down_t=QT::Q6_K;
  std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ud(0,1);
  static std::vector<float> an(hidden), fn(hidden), cw((size_t)conv_dim*kernel), A(vh), dt(vh), gn(hd);
  for (auto& v:an) v=0.5f+ud(rng); for (auto& v:fn) v=0.5f+ud(rng); for (auto& v:cw) v=nd(rng)*0.4f; for (auto& v:A) v=-(0.05f+ud(rng)); for (auto& v:dt) v=nd(rng)*0.3f; for (auto& v:gn) v=0.5f+ud(rng);
  d.attn_norm=an.data(); d.ffn_norm=fn.data(); d.conv_w=cw.data(); d.A=A.data(); d.dt=dt.data(); d.gnorm=gn.data();

  // reference state (double) and the host copies handed to the backend for the first upload
  std::vector<double> cst((size_t)conv_dim*(kernel-1),0.0), rst((size_t)vh*hd*hd,0.0);
  for (auto& v:cst) v=nd(rng)*0.2; for (auto& v:rst) v=nd(rng)*0.05;
  std::vector<float> cst_h(cst.begin(),cst.end()), rst_h(rst.begin(),rst.end());
  std::vector<double> xr(hidden); std::vector<float> x(hidden);
  int bad=0; const int ntok=8;
  for (int t = 0; t < ntok; ++t) {
    for (uint32_t i=0;i<hidden;i++){ xr[i]=nd(rng); x[i]=(float)xr[i]; }
    // ---- reference ----
    std::vector<double> xb = rms(xr,an,eps), mixed,z,a,b; mv(dq,conv_dim,hidden,xb,mixed); mv(dz,value_dim,hidden,xb,z); mv(da,vh,hidden,xb,a); mv(db,vh,hidden,xb,b);
    std::vector<double> co(conv_dim);
    for (uint32_t c=0;c<conv_dim;c++){ double s=0; for(uint32_t k=0;k<kernel;k++){ double xv=(k<kernel-1)?cst[(size_t)c*(kernel-1)+k]:mixed[c]; s+=xv*cw[(size_t)c*kernel+k]; } co[c]=s/(1.0+std::exp(-s)); }
    for (uint32_t c=0;c<conv_dim;c++){ double* st=&cst[(size_t)c*(kernel-1)]; for(uint32_t k=0;k+2<kernel;k++) st[k]=st[k+1]; st[kernel-2]=mixed[c]; }
    std::vector<double> qn(co.begin(),co.begin()+key_dim), kn(co.begin()+key_dim,co.begin()+2*key_dim);
    for (uint32_t h=0;h<kh;h++){ double s1=0,s2=0; for(uint32_t i=0;i<hd;i++){ s1+=qn[h*hd+i]*qn[h*hd+i]; s2+=kn[h*hd+i]*kn[h*hd+i]; } for(uint32_t i=0;i<hd;i++){ qn[h*hd+i]/=std::sqrt(s1+1e-6); kn[h*hd+i]/=std::sqrt(s2+1e-6); } }
    std::vector<double> core(value_dim);
    for (uint32_t hv=0;hv<vh;hv++){ uint32_t hk=hv/(vh/kh); double* S=&rst[(size_t)hv*hd*hd]; double beta=sg(b[hv]); double sp=a[hv]+dt[hv]; double spv=sp>40?sp:(sp<-40?std::exp(sp):std::log1p(std::exp(sp))); double gg=A[hv]*spv; if(gg>0)gg=0; if(gg<-60)gg=-60; double dec=std::exp(gg);
      for(uint32_t i=0;i<hd*hd;i++) S[i]*=dec; std::vector<double> kvm(hd),del(hd),raw(hd);
      for(uint32_t j=0;j<hd;j++){ double s=0; for(uint32_t i=0;i<hd;i++) s+=kn[hk*hd+i]*S[(size_t)i*hd+j]; kvm[j]=s; del[j]=(co[2*key_dim+hv*hd+j]-s)*beta; }
      for(uint32_t i=0;i<hd;i++) for(uint32_t j=0;j<hd;j++) S[(size_t)i*hd+j]+=kn[hk*hd+i]*del[j];
      double ss=0; for(uint32_t j=0;j<hd;j++){ double s=0; for(uint32_t i=0;i<hd;i++) s+=qn[hk*hd+i]/std::sqrt((double)hd)*S[(size_t)i*hd+j]; raw[j]=s; ss+=s*s; }
      double inv=1.0/std::sqrt(ss/hd+eps); for(uint32_t j=0;j<hd;j++){ double zi=z[hv*hd+j]; core[hv*hd+j]=raw[j]*inv*gn[j]*(zi/(1.0+std::exp(-zi))); } }
    std::vector<double> o; mv(dout,hidden,value_dim,core,o); for(uint32_t i=0;i<hidden;i++) xr[i]+=o[i];
    std::vector<double> xn=rms(xr,fn,eps), gt,up,hm(ffn),dn; mv(dg,ffn,hidden,xn,gt); mv(du,ffn,hidden,xn,up); for(uint32_t i=0;i<ffn;i++) hm[i]=gt[i]/(1.0+std::exp(-gt[i]))*up[i];
    mv(dd,hidden,ffn,hm,dn); for(uint32_t i=0;i<hidden;i++) xr[i]+=dn[i];
    // ---- backend ----
    bool ok = g.q35_linear_layer(d, x.data(), cst_h.data(), rst_h.data());
    double e = ok ? relerr(x, xr) : 1.0; bool pass = ok && e < 2e-3; if (!pass) bad++;
    printf("token %d  resident-before=%d  rel err %.3e  %s\n", t, 0, e, pass?"PASS":"FAIL");
    if (t == 3) {   // drop the card copy, bring the state back to the host, and continue from the host copies
      std::vector<float> c2(cst_h.size()), r2(rst_h.size()); bool dl = g.q35_state_download(&g, 3, c2.data(), r2.data());
      double ec=0,er=0,sc=0,sr=0; for(size_t i=0;i<c2.size();i++){ ec+=(c2[i]-cst[i])*(c2[i]-cst[i]); sc+=cst[i]*cst[i]; } for(size_t i=0;i<r2.size();i++){ er+=(r2[i]-rst[i])*(r2[i]-rst[i]); sr+=rst[i]*rst[i]; }
      printf("   state download: ok=%d conv rel err %.2e  rec rel err %.2e\n", dl, std::sqrt(ec/std::max(sc,1e-30)), std::sqrt(er/std::max(sr,1e-30)));
      if(!dl||std::sqrt(ec/sc)>1e-3||std::sqrt(er/sr)>1e-3) bad++;
      g.q35_state_drop(&g, 3); printf("   dropped; resident=%d (expect 0)\n", g.q35_state_resident(&g,3)); if (g.q35_state_resident(&g,3)) bad++;
      cst_h=c2; rst_h=r2;
    }
  }
  // invalid description must be rejected without touching x
  { auto bd = d; bd.hd = 96; std::vector<float> x0(x); bool ok = g.q35_linear_layer(bd, x.data(), cst_h.data(), rst_h.data()); printf("invalid desc rejected: %s\n", (!ok && x0==x) ? "yes":"NO"); if (ok || x0!=x) bad++; }
  printf(bad ? "FAILED %d\n" : "all passed\n", bad); return bad?1:0;
}
