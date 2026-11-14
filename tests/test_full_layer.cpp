// Runs VulkanBackend::q35_full_layer (the real backend + shaders) against a plain CPU reference.
#include "gpu/vulkan_backend.h"
#include <random>
#include <cstdio>
#include <cmath>
using namespace nexus; using quant::QuantType;
static float h2f(uint16_t h){ uint32_t s=(h&0x8000u)<<16,e=(h>>10)&31,m=h&1023; uint32_t x;
  if(!e){ if(!m) x=s; else { e=1; while(!(m&1024)){m<<=1;--e;} m&=1023; x=s|((e+112)<<23)|(m<<13);} } else if(e==31) x=s|0x7F800000|(m<<13); else x=s|((e+112)<<23)|(m<<13);
  float f; memcpy(&f,&x,4); return f; }
static uint16_t f2h(float f){ _Float16 h=(_Float16)f; uint16_t o; memcpy(&o,&h,2); return o; }
struct Case { const char* name; int hidden, nh, nkv, hd, rot, ffn; bool gate, qn, kn, bias, inter; };
int main(){
  gpu::VulkanBackend g; if(!g.init(false)){ printf("no gpu\n"); return 2; }
  std::mt19937 rng(3); std::uniform_real_distribution<float> U(-1,1);
  const Case cases[] = {
    {"qwen3.5 gated (unchanged path)", 512, 8, 2, 128, 32, 768, true,  true,  true,  false, false},
    {"llama/mistral plain, interleaved", 512, 8, 2, 128, 128, 768, false, false, false, false, true},
    {"qwen2.5 (biases, G=7)",          448, 7, 1, 64, 64, 640, false, false, false, true,  false},
    {"qwen2.5-7B shape (G=7, hd=128)", 896, 7, 1, 128, 128, 1024, false, false, false, true, false},
    {"qwen3 dense (q/k norm, no gate)", 512, 8, 4, 64, 64, 768, false, true,  true,  false, false},
  };
  int fails = 0; std::vector<std::vector<float>> keep;
  auto vec=[&](size_t n, float s){ keep.emplace_back(n); for(auto&v:keep.back()) v=U(rng)*s; return keep.back().data(); };
  int owner_tag = 0;
  for (const Case& c : cases) {
    const int H=c.hidden, QD=c.nh*c.hd, KD=c.nkv*c.hd, F=c.ffn, G=c.nh/c.nkv;
    g.init_activation_bufs(H, 2*QD, KD, F);
    const float ws = 1.0f/std::sqrt((float)H);
    float* an=vec(H,0.2f); for(int i=0;i<H;++i) an[i]+=1.f; float* fn=vec(H,0.2f); for(int i=0;i<H;++i) fn[i]+=1.f;
    float* wq=vec((size_t)(c.gate?2:1)*QD*H,ws); float* wk=vec((size_t)KD*H,ws); float* wv=vec((size_t)KD*H,ws); float* wo=vec((size_t)H*QD,1.f/std::sqrt((float)QD));
    float* wg=vec((size_t)F*H,ws); float* wu=vec((size_t)F*H,ws); float* wd=vec((size_t)H*F,1.f/std::sqrt((float)F));
    float* qn=c.qn?vec(c.hd,0.3f):nullptr; float* kn=c.kn?vec(c.hd,0.3f):nullptr;
    if(qn) for(int i=0;i<c.hd;++i) qn[i]+=1.f; if(kn) for(int i=0;i<c.hd;++i) kn[i]+=1.f;
    float* bq=c.bias?vec(QD,0.5f):nullptr; float* bk=c.bias?vec(KD,0.5f):nullptr; float* bv=c.bias?vec(KD,0.5f):nullptr; float* bo=c.bias?vec(H,0.1f):nullptr;
    // host KV cache with P earlier tokens (already roped, f16)
    const int P = 300;
    std::vector<std::vector<uint16_t>> Kc(c.nkv, std::vector<uint16_t>((size_t)(P+1)*c.hd)), Vc=Kc;
    for(auto& kk: Kc) for(auto& v: kk) v=f2h(U(rng)); for(auto& vv: Vc) for(auto& v: vv) v=f2h(U(rng));
    const uint16_t* kp[16]; const uint16_t* vp[16]; for(int h=0;h<c.nkv;++h){ kp[h]=Kc[h].data(); vp[h]=Vc[h].data(); }
    // rope table (rotate_half probe form): cos[i], sin[i] for pair i, freq = 1e4^(-2i/rot)
    std::vector<float> tab(c.rot); for(int i=0;i<c.rot/2;++i){ double th=P*std::pow(1e4,-2.0*i/c.rot); tab[i]=(float)cos(th); tab[c.rot/2+i]=(float)sin(th); }
    std::vector<float> x(H); for(auto& v: x) v=U(rng);
    // ---- CPU reference
    auto rms=[&](const float* in,const float* w,int n,std::vector<double>& out){ double ss=0; for(int i=0;i<n;++i) ss+=(double)in[i]*in[i]; double inv=1/std::sqrt(ss/n+1e-6); out.resize(n); for(int i=0;i<n;++i) out[i]=in[i]*inv*w[i]; };
    auto mv=[&](const float* W,const std::vector<double>& in,int rows,int cols){ std::vector<double> o(rows); for(int r=0;r<rows;++r){ double s=0; for(int k=0;k<cols;++k) s+=(double)W[(size_t)r*cols+k]*in[k]; o[r]=s;} return o; };
    std::vector<double> xb; rms(x.data(),an,H,xb);
    auto qg=mv(wq,xb,(c.gate?2:1)*QD,H), kr=mv(wk,xb,KD,H), vr=mv(wv,xb,KD,H);
    std::vector<double> q(QD), gate(QD,1e9);
    for(int h=0;h<c.nh;++h) for(int d=0;d<c.hd;++d){ if(c.gate){ q[h*c.hd+d]=qg[h*2*c.hd+d]; gate[h*c.hd+d]=qg[h*2*c.hd+c.hd+d]; } else q[h*c.hd+d]=qg[h*c.hd+d]; }
    if(bq) for(int i=0;i<QD;++i) q[i]+=bq[i]; if(bk) for(int i=0;i<KD;++i) kr[i]+=bk[i]; if(bv) for(int i=0;i<KD;++i) vr[i]+=bv[i];
    auto headnorm_rope=[&](double* v,const float* w,bool zc){ if(w){ double ss=0; for(int d=0;d<c.hd;++d) ss+=v[d]*v[d]; double inv=1/std::sqrt(ss/c.hd+1e-6); for(int d=0;d<c.hd;++d) v[d]=v[d]*inv*(w[d]+(zc?1:0)); }
      std::vector<double> o(v,v+c.hd); int hr=c.rot/2;
      for(int i=0;i<hr;++i){ double cs=tab[i], sn=tab[hr+i];
        if(c.inter){ o[2*i]=v[2*i]*cs-v[2*i+1]*sn; o[2*i+1]=v[2*i]*sn+v[2*i+1]*cs; }
        else { o[i]=v[i]*cs-v[i+hr]*sn; o[i+hr]=v[i+hr]*cs+v[i]*sn; } }
      for(int d=0;d<c.hd;++d) v[d]=o[d]; };
    for(int h=0;h<c.nh;++h) headnorm_rope(&q[h*c.hd],qn,false);
    for(int h=0;h<c.nkv;++h) headnorm_rope(&kr[h*c.hd],kn,false);
    std::vector<double> att(QD);
    for(int h=0;h<c.nh;++h){ int kvh=h/G; std::vector<double> sc(P+1); double mx=-1e300;
      for(int t=0;t<=P;++t){ double s=0; for(int d=0;d<c.hd;++d){ double kk = t<P ? h2f(Kc[kvh][(size_t)t*c.hd+d]) : h2f(f2h((float)kr[kvh*c.hd+d])); s+=q[h*c.hd+d]*kk; } sc[t]=s/std::sqrt((double)c.hd); mx=std::max(mx,sc[t]); }
      double L=0; for(auto& s: sc){ s=std::exp(s-mx); L+=s; }
      for(int d=0;d<c.hd;++d){ double a=0; for(int t=0;t<=P;++t){ double vv = t<P ? h2f(Vc[kvh][(size_t)t*c.hd+d]) : h2f(f2h((float)vr[kvh*c.hd+d])); a+=sc[t]*vv; }
        double sg = c.gate ? 1/(1+std::exp(-gate[h*c.hd+d])) : 1.0; att[h*c.hd+d]=a/L*sg; } }
    auto o=mv(wo,att,H,QD); std::vector<double> x1(H); for(int i=0;i<H;++i) x1[i]=x[i]+o[i]+(bo?bo[i]:0);
    std::vector<float> x1f(x1.begin(),x1.end()); std::vector<double> xb2; rms(x1f.data(),fn,H,xb2);
    auto gg=mv(wg,xb2,F,H), uu=mv(wu,xb2,F,H); std::vector<double> act(F); for(int i=0;i<F;++i) act[i]=gg[i]/(1+std::exp(-gg[i]))*uu[i];
    auto dn=mv(wd,act,H,F); std::vector<double> ref(H); for(int i=0;i<H;++i) ref[i]=x1[i]+dn[i];
    // ---- GPU
    gpu::VulkanBackend::Q35FullDesc d;
    d.owner=&owner_tag; d.layer=(int)(&c-cases); d.hidden=H; d.n_heads=c.nh; d.n_kv=c.nkv; d.hd=c.hd; d.rot=c.rot; d.ffn=F; d.pos=P; d.norm_eps=1e-6f; d.scale=0.f;
    d.attn_norm=an; d.ffn_norm=fn; d.wq=wq; d.wq_t=QuantType::F32; d.wk=wk; d.wk_t=QuantType::F32; d.wv=wv; d.wv_t=QuantType::F32; d.wo=wo; d.wo_t=QuantType::F32;
    d.q_norm=qn; d.k_norm=kn; d.rope_tab=tab.data(); d.gate=wg; d.gate_t=QuantType::F32; d.up=wu; d.up_t=QuantType::F32; d.down=wd; d.down_t=QuantType::F32;
    d.has_gate=c.gate; d.rope_interleaved=c.inter; d.bq=bq; d.bk=bk; d.bv=bv; d.bo=bo;
    d.k_host=kp; d.v_host=vp; d.host_tokens=P+1; d.host_synced=0;
    std::vector<float> xg=x, kvo(2*KD); uint32_t mv_=0;
    bool ok=g.q35_full_layer(d, xg.data(), kvo.data(), &mv_);
    double e=0, rm=0; for(int i=0;i<H;++i){ e=std::max(e,std::fabs(xg[i]-ref[i])); rm=std::max(rm,std::fabs(ref[i]-x[i])); }
    double ek=0; for(int i=0;i<KD;++i) ek=std::max(ek,std::fabs(kvo[i]-kr[i]));
    const bool bad = !ok || !(e < 2e-2*std::max(1.0,rm)) || !(ek < 1e-2);
    printf("%-34s %s  layer out err %.2e (scale %.2f)  new K err %.2e  %s\n", c.name, ok?"gpu":"DECLINED", e, rm, ek, bad?"FAIL":"ok");
    if(bad) ++fails;
  }
  printf(fails ? "RESULT: %d failure(s)\n" : "RESULT: all ok\n", fails); return fails;
}
