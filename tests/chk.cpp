#include "/tmp/chk/src/gguf/quant.h"
#include <cstdio>
#include <vector>
#include <cmath>
using namespace nexus::quant;
static std::vector<uint8_t> rd(const char* f){ FILE* p=fopen(f,"rb"); std::vector<uint8_t> v; uint8_t b[4096]; size_t n; while((n=fread(b,1,4096,p))>0) v.insert(v.end(),b,b+n); fclose(p); return v; }
int main(){
  int bad=0;
  for (auto nm : {"q2k","q3k"}) {
    auto blk = rd((std::string(nm)+".blocks").c_str()); auto tr = rd((std::string(nm)+".truth").c_str());
    const float* t = (const float*)tr.data(); size_t n = tr.size()/4; std::vector<float> out(n);
    QuantType qt = std::string(nm)=="q2k" ? QuantType::Q2_K : QuantType::Q3_K;
    dequantize(blk.data(), out.data(), n, qt);
    double mx=0; for (size_t i=0;i<n;i++) mx=std::max(mx,(double)std::fabs(out[i]-t[i]));
    printf("%s: %zu values, max abs diff vs gguf-py = %g\n", nm, n, mx); if (mx>1e-6) bad=1;
  }
  return bad;
}
