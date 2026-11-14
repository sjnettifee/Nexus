#include "src/cpu/matmul.h"
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
using namespace nexus; using namespace nexus::cpu;
int main(int argc,char**argv){bool i8=argc>1;
 size_t rows=17408,cols=5120,bpr=cols/64; std::vector<quant::BlockNQ_G64<4>> W(rows*bpr);std::mt19937 r(1);std::normal_distribution<float>N(0,1);
 for(auto&b:W){b.d=(uint16_t)(0x2000+(r()%64));for(auto&q:b.qs)q=r();}
 std::vector<float>X(cols*4),Y(rows*4),Yref(rows*4);for(auto&x:X)x=N(r)*(1+0.5f*((r()%10)==0?4:0));
 auto T=[&](auto f){f();double best=1e9;for(int k=0;k<5;k++){auto a=std::chrono::steady_clock::now();f();best=std::min(best,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count());}return best;};
 double t1=T([&]{mat_vec(W.data(),X.data(),Y.data(),rows,cols,QuantType::NQ4_G64,2);});
 double t4=T([&]{mat_mat(W.data(),X.data(),Y.data(),rows,cols,4,QuantType::NQ4_G64,2);});
 printf("%s: 1 token %.1f ms, 4 tokens %.1f ms\n", i8?"8-bit":"float", t1, t4);
 // accuracy: compare with an exact double reference
 mat_vec(W.data(),X.data(),Y.data(),rows,cols,QuantType::NQ4_G64,2);
 double num=0,den=0,mx=0;for(size_t row=0;row<rows;row++){double ref=0;for(size_t b=0;b<bpr;b++){double d=quant::f16_to_f32(W[row*bpr+b].d);for(int j=0;j<32;j++){uint8_t by=W[row*bpr+b].qs[j];ref+=d*(((int)(by&15))-8)*X[b*64+2*j]+d*(((int)(by>>4))-8)*X[b*64+2*j+1];}}
  double e=Y[row]-ref;num+=e*e;den+=ref*ref;mx=std::max(mx,fabs(e)/ (sqrt(den/(row+1))+1e-30));}
 printf("   error vs exact: rms relative %.2e\n", sqrt(num/den));
 // mat_mat consistency with mat_vec
 mat_mat(W.data(),X.data(),Y.data(),rows,cols,4,QuantType::NQ4_G64,2);
 for(int t=0;t<4;t++)mat_vec(W.data(),X.data()+t*cols,Yref.data()+t*rows,rows,cols,QuantType::NQ4_G64,2);
 double md=0;for(size_t i=0;i<Y.size();i++)md=std::max(md,(double)fabsf(Y[i]-Yref[i]));printf("   batch vs single max diff %.2e\n",md);}
