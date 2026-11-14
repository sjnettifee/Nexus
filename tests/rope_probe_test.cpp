#include "../src/model/attention.h"
#include <random>
using namespace nexus::model;
int main(){
  std::mt19937 rng(3); std::normal_distribution<float> nd(0,1); int bad=0;
  for (int type=0; type<3; ++type) for (int pos : {0, 1, 7, 1000, 49999}) {
    RoPEConfig rc; rc.freq_base=1.0e7f; rc.scale = type==0?1.0f:4.0f; rc.type=type; rc.n_ctx_orig=8192; rc.layout=1;
    const int rot=64, hd=256; std::vector<float> x(hd); for(auto& v:x) v=nd(rng); std::vector<float> ref=x;
    rope_apply(ref.data(), pos, rot, rc);                       // what the CPU path does to the first rot dims
    std::vector<float> probe(rot,0.f); for(int i=0;i<rot/2;i++) probe[i]=1.f; rope_apply(probe.data(), pos, rot, rc);
    std::vector<float> y=x; const int half=rot/2;
    for(int i=0;i<half;i++){ float c=probe[i], s=probe[half+i]; y[i]=x[i]*c-x[i+half]*s; y[i+half]=x[i+half]*c+x[i]*s; }
    double e=0; for(int i=0;i<rot;i++) e=std::max(e,(double)std::fabs(y[i]-ref[i])); if(e>1e-6) bad++;
    printf("type %d pos %5d  max diff %.2e\n", type, pos, e);
  }
  printf(bad?"FAILED\n":"all passed\n"); return bad;
}
