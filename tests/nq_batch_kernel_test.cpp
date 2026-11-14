#include "src/cpu/matmul.h"
#include <random>
#include <chrono>
#include <cstdio>
using namespace nexus; using namespace nexus::cpu;
template<int B> static std::vector<uint8_t> mk(size_t rows,size_t cols,std::mt19937&r){size_t bpr=(cols+63)/64;std::vector<uint8_t> v(rows*bpr*sizeof(quant::BlockNQ_G64<B>));auto*b=(quant::BlockNQ_G64<B>*)v.data();std::uniform_int_distribution<int>u(0,255);for(size_t i=0;i<rows*bpr;i++){b[i].d=(uint16_t)(0x2000+(i%7)*37);for(auto&q:b[i].qs)q=(uint8_t)u(r);}return v;}
template<int B> static int test(QuantType qt,size_t rows,size_t cols,int ntok,int thr,bool timeit){
 std::mt19937 r(1);auto W=mk<B>(rows,cols,r);std::normal_distribution<float>N(0,1);std::vector<float>X(cols*ntok);for(auto&x:X)x=N(r);
 std::vector<float>Y1(rows*ntok),Y2(rows*ntok);
 for(int t=0;t<ntok;t++)mat_vec(W.data(),X.data()+t*cols,Y1.data()+t*rows,rows,cols,qt,thr);
 mat_mat(W.data(),X.data(),Y2.data(),rows,cols,ntok,qt,thr);
 size_t bad=0;double mr=0,ms=0;for(size_t i=0;i<Y1.size();i++){if(memcmp(&Y1[i],&Y2[i],4))bad++;ms=std::max(ms,(double)fabsf(Y1[i]));mr=std::max(mr,(double)fabsf(Y1[i]-Y2[i]));}
 double rel=mr/std::max(1e-30,ms);
 printf("NQ%d %zux%zu ntok=%d: %zu of %zu differ, max diff %.2e of max |y| %.2e (rel %.1e)\n",B,rows,cols,ntok,bad,Y1.size(),mr,ms,rel);
 if(rel>1e-5) return 1;
 if(timeit){auto T=[&](auto f){f();auto a=std::chrono::steady_clock::now();for(int k=0;k<3;k++)f();return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count()/3;};
  double t1=T([&]{mat_vec(W.data(),X.data(),Y1.data(),rows,cols,qt,thr);});
  double t4=T([&]{mat_mat(W.data(),X.data(),Y2.data(),rows,cols,ntok,qt,thr);});
  printf("   time: 1 token %.1f ms, batch of %d %.1f ms = %.2fx one token\n",t1,ntok,t4,t4/t1);}
 return (B!=4 && bad!=0);}
int main(){int f=0;
 f+=test<4>(QuantType::NQ4_G64,17408,5120,4,2,true);   // FFN gate/up of the 27B
 f+=test<4>(QuantType::NQ4_G64,5120,17408,4,2,true);   // FFN down
 f+=test<4>(QuantType::NQ4_G64,333,250,3,2,false);     // odd sizes (tail blocks)
 f+=test<5>(QuantType::NQ5_G64,1000,5120,4,2,false);
 f+=test<6>(QuantType::NQ6_G64,1000,5120,4,2,false);
 f+=test<8>(QuantType::NQ8_G64,1000,5120,4,2,false);
 printf("FAILURES: %d\n",f);return f;}
