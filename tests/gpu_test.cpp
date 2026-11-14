// gpu_test.cpp -- run the REAL Vulkan backend on whatever device is selected and compare every
// supported quant type against the CPU kernels / a double-precision reference.
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ./gpu_test [big]
#include "gpu/vulkan_backend.h"
#include "cpu/matmul.h"
#include <cstdio>
#include <random>
#include <cmath>
using namespace nexus;
using quant::QuantType;

static uint16_t f32_to_f16(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t s=(x>>16)&0x8000; int e=((x>>23)&0xFF)-127+15; if(e<=0)return s; if(e>=31)return s|0x7BFF; return s|(e<<10)|((x&0x7FFFFF)>>13);}
struct T{const char*name;QuantType t;size_t be,bb;double tol;};
static void fill(std::vector<uint8_t>&W,const T&t,size_t rows,size_t cols,std::mt19937&rng){
  size_t rb=(cols/t.be)*t.bb; W.resize(rb*rows);
  for(auto&b:W)b=(uint8_t)(rng()&0xFF);
  auto setd=[&](size_t o){uint16_t h=f32_to_f16(0.01f+0.02f*(float)(rng()%100)/100.f);memcpy(&W[o],&h,2);};
  for(size_t r=0;r<rows;++r)for(size_t b=0;b<cols/t.be;++b){size_t o=r*rb+b*t.bb;
    switch(t.t){case QuantType::F32:{float f=(float)(rng()%2000)/1000.f-1.f;memcpy(&W[o],&f,4);break;}
      case QuantType::F16:{uint16_t h=f32_to_f16((float)(rng()%2000)/1000.f-1.f+2.f);memcpy(&W[o],&h,2);break;}
      case QuantType::Q4_K:case QuantType::Q5_K:setd(o);setd(o+2);break;
      case QuantType::Q6_K:setd(o+208);break; default:setd(o);}}
}
int main(int argc,char**argv){
  bool big=argc>1&&std::string(argv[1])=="big";
  gpu::VulkanBackend gb;
  if(!gb.init(true)){printf("GPU init failed\n");return 2;}
  const T types[]={{"F32",QuantType::F32,1,4,1e-4},{"F16",QuantType::F16,1,2,1e-3},
    {"Q8_0",QuantType::Q8_0,32,34,1e-3},{"Q4_K",QuantType::Q4_K,256,144,4e-2},{"Q5_K",QuantType::Q5_K,256,176,4e-2},
    {"Q6_K",QuantType::Q6_K,256,210,4e-2},{"NQ4",QuantType::NQ4_G64,64,34,1e-3},{"NQ5",QuantType::NQ5_G64,64,42,1e-3},
    {"NQ6",QuantType::NQ6_G64,64,50,1e-3},{"NQ8",QuantType::NQ8_G64,64,66,1e-3}};
  std::mt19937 rng(7); int fails=0; std::vector<std::vector<uint8_t>> keep; // weights are cached by pointer: keep every buffer alive and unique
  const size_t ROWS=big?151936:131, COLS=big?5120:1024, NTOK=big?3:19;
  printf("%-5s %-9s %-13s %-13s\n","type","backend","mat_vec err","mat_mat err");
  for(const T&t:types){
    if(big && t.t!=QuantType::NQ4_G64) continue;
    keep.emplace_back(); std::vector<uint8_t>&W=keep.back(); fill(W,t,ROWS,COLS,rng);
    std::vector<float>X(COLS*NTOK); for(auto&v:X)v=(float)(rng()%2000)/1000.f-1.f;
    // CPU reference (double)
    std::vector<double>ref(ROWS*NTOK); std::vector<float>deq(COLS); double rmax=0;
    for(size_t r=0;r<ROWS;++r){quant::dequantize_row(W.data(),deq.data(),r,COLS,t.t);
      for(size_t k=0;k<NTOK;++k){double s=0;for(size_t c=0;c<COLS;++c)s+=(double)deq[c]*X[k*COLS+c];ref[k*ROWS+r]=s;rmax=std::max(rmax,std::fabs(s));}}
    auto err=[&](const std::vector<float>&y){double e=0;for(size_t i=0;i<y.size();++i)e=std::max(e,std::fabs((double)y[i]-ref[i]));return e/(rmax+1e-9);};
    std::vector<float> yv(ROWS*NTOK,-777.f), ym(ROWS*NTOK,-777.f);
    bool okv=true; for(size_t k=0;k<NTOK;++k) okv&=gb.mat_vec(W.data(),&X[k*COLS],&yv[k*ROWS],ROWS,COLS,t.t);
    bool okm=gb.mat_mat_supported(t.t)&&gb.mat_mat(W.data(),X.data(),ym.data(),ROWS,COLS,NTOK,t.t);
    double ev=okv?err(yv):-1, em=okm?err(ym):-1;
    bool bad=(okv&&(!(ev<t.tol)))||(okm&&(!(em<t.tol)));
    printf("%-5s v:%-3s m:%-3s %-13.3e %-13.3e %s\n",t.name,okv?"gpu":"cpu",okm?"gpu":"cpu",ev,em,bad?"FAIL":"ok");
    if(bad)++fails;
  }
  printf(fails?"RESULT: %d FAILURE(S)\n":"RESULT: all ok\n",fails); return fails?1:0;
}
