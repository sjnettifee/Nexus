#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
static float sig(float x){return 1.f/(1.f+expf(-x));}
static float sp(float x){return x>20?x:log1pf(expf(x));}
static void l2n(float*x,int n,float eps){float s=0;for(int i=0;i<n;i++)s+=x[i]*x[i];float r=1.f/sqrtf(s+eps);for(int i=0;i<n;i++)x[i]*=r;}
struct L{int conv_dim,key_dim,value_dim,kh,vh,hd,kernel;std::vector<float> conv,A,dt;};
// ORIGINAL: CPU conv loop + scan state update copied from forward_qwen35_linear_batch (outputs dropped)
static void orig(const L&lw,bool grouped,const float*mixed,const float*a,const float*b,int n,std::vector<float>&cs,std::vector<float>&rs){
 int conv_dim=lw.conv_dim,kernel=lw.kernel,hd=lw.hd,kh=lw.kh,vh=lw.vh,key_dim=lw.key_dim,repeat=vh/kh;
 std::vector<float> conv_out((size_t)n*conv_dim);
 for(int t=0;t<n;++t){const float*mixt=mixed+(size_t)t*conv_dim;float*convt=conv_out.data()+(size_t)t*conv_dim;
  for(int c=0;c<conv_dim;++c){float sum=0;for(int k=0;k<kernel;++k){float xv;if(k<kernel-1)xv=cs[(size_t)c*(kernel-1)+k];else xv=mixt[c];sum+=xv*lw.conv[(size_t)c*kernel+k];}convt[c]=sum/(1.f+expf(-sum));}
  if(kernel>1)for(int c=0;c<conv_dim;++c){float*st=cs.data()+(size_t)c*(kernel-1);for(int k=0;k<kernel-2;++k)st[k]=st[k+1];st[kernel-2]=mixt[c];}}
 std::vector<float> q_flat(key_dim),k_flat(key_dim),v_flat((size_t)vh*hd),q_normed(key_dim),k_normed(key_dim);
 for(int t=0;t<n;++t){const float*convt=conv_out.data()+(size_t)t*conv_dim;const float*q_base,*k_base,*v_base;
  if(grouped){int gs=2*hd+repeat*hd;for(int hk=0;hk<kh;++hk){size_t src=(size_t)hk*gs;memcpy(q_flat.data()+hk*hd,convt+src,hd*4);memcpy(k_flat.data()+hk*hd,convt+src+hd,hd*4);for(int r=0;r<repeat;++r){size_t dv=((size_t)hk*repeat+r)*hd,sv=src+2*hd+(size_t)r*hd;memcpy(v_flat.data()+dv,convt+sv,hd*4);}}q_base=q_flat.data();k_base=k_flat.data();v_base=v_flat.data();}
  else{q_base=convt;k_base=q_base+key_dim;v_base=k_base+key_dim;}
  memcpy(q_normed.data(),q_base,key_dim*4);memcpy(k_normed.data(),k_base,key_dim*4);
  for(int hk=0;hk<kh;++hk){l2n(q_normed.data()+hk*hd,hd,1e-6f);l2n(k_normed.data()+hk*hd,hd,1e-6f);}
  const float*at=a+(size_t)t*vh,*bt=b+(size_t)t*vh;
  for(int hv=0;hv<vh;++hv){int hk=hv/repeat;const float*k=k_normed.data()+hk*hd;const float*v=v_base+(size_t)hv*hd;
   float beta=sig(bt[hv]);float g=lw.A[hv]*sp(at[hv]+lw.dt[hv]);if(!std::isfinite(g))g=-60;if(g>0)g=0;if(g<-60)g=-60;float decay=expf(g);
   float*S=rs.data()+(size_t)hv*hd*hd;for(int i=0;i<hd*hd;++i)S[i]*=decay;float kv_mem[256];for(int j=0;j<hd;++j){float s=0;for(int i=0;i<hd;++i)s+=k[i]*S[(size_t)i*hd+j];kv_mem[j]=s;}
   float delta[256];for(int j=0;j<hd;++j)delta[j]=(v[j]-kv_mem[j])*beta;for(int i=0;i<hd;++i){float ki=k[i];float*Sr=S+(size_t)i*hd;for(int j=0;j<hd;++j)Sr[j]+=ki*delta[j];}}}
}
// REPLAY: body of Transformer::q35_replay_state (copied verbatim below by the build script)
#include "replay_body.inc"
int main(){std::mt19937 rng(7);std::normal_distribution<float>N(0,1);int fails=0;
 for(int grouped=0;grouped<2;++grouped)for(int keep=1;keep<=4;++keep){
  L lw;lw.hd=16;lw.kh=2;lw.vh=4;lw.kernel=4;lw.key_dim=lw.kh*lw.hd;lw.value_dim=lw.vh*lw.hd;lw.conv_dim=2*lw.key_dim+lw.value_dim;
  lw.conv.resize((size_t)lw.conv_dim*lw.kernel);for(auto&x:lw.conv)x=0.3f*N(rng);lw.A.resize(lw.vh);lw.dt.resize(lw.vh);for(auto&x:lw.A)x=-fabsf(N(rng));for(auto&x:lw.dt)x=N(rng);
  int n=4;std::vector<float> mixed((size_t)n*lw.conv_dim),a((size_t)n*lw.vh),b((size_t)n*lw.vh);for(auto&x:mixed)x=N(rng);for(auto&x:a)x=N(rng);for(auto&x:b)x=N(rng);
  std::vector<float> cs0((size_t)lw.conv_dim*(lw.kernel-1)),rs0((size_t)lw.vh*lw.hd*lw.hd);for(auto&x:cs0)x=N(rng);for(auto&x:rs0)x=0.1f*N(rng);
  auto cs1=cs0,rs1=rs0;orig(lw,grouped,mixed.data(),a.data(),b.data(),keep,cs1,rs1);
  auto cs2=cs0,rs2=rs0;replay(lw,grouped,mixed.data(),a.data(),b.data(),keep,cs2,rs2);
  double md=0;for(size_t i=0;i<cs1.size();++i)md=std::max(md,(double)fabsf(cs1[i]-cs2[i]));for(size_t i=0;i<rs1.size();++i)md=std::max(md,(double)fabsf(rs1[i]-rs2[i]));
  // also: replay(keep) must differ from the full 4-token state when keep<4 (sanity: it really stops)
  auto cs3=cs0,rs3=rs0;orig(lw,grouped,mixed.data(),a.data(),b.data(),4,cs3,rs3);double dfull=0;for(size_t i=0;i<rs1.size();++i)dfull=std::max(dfull,(double)fabsf(rs3[i]-rs2[i]));
  bool ok=md<1e-5&&(keep==4?dfull<1e-5:dfull>1e-3);if(!ok)++fails;
  printf("%s keep=%d max diff vs original=%.2e  (vs full batch %.2e) %s\n",grouped?"grouped":"flat   ",keep,md,dfull,ok?"OK":"FAIL");}
 printf("FAILURES: %d\n",fails);return fails;}
