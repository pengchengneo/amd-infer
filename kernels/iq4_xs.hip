#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <cstdlib>
#include <initializer_list>
#include "ggml_reference/dequant.h"
#include "execution_graph.h"
#include "gdn_profile.h"
#include "attention_profile.h"
// Wire layout and non-linear codebook compatible with GGML (MIT), see THIRD_PARTY.md.
struct Block { uint16_t d, scales_h; uint8_t scales_l[4], qs[128]; };
static_assert(sizeof(Block)==136);
__device__ __constant__ float levels[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
__device__ float inline_level(int code){
    uint64_t packed=code<8?0xf6eaddcfbfad9881ULL:0x7159453526190d01ULL;
    int value=int((packed>>(8*(code&7)))&255);return float(value>=128?value-256:value);
}
__device__ float weight(const Block& b,int i,bool inline_levels=false){
    int s=i/32,j=i%32;int sc=((b.scales_l[s/2]>>(4*(s%2)))&15)|(((b.scales_h>>(2*s))&3)<<4);
    uint8_t q=b.qs[s*16+j%16];int code=j<16?(q&15):(q>>4);
    return ai_half(b.d)*(sc-32)*(inline_levels?inline_level(code):levels[code]);
}
template<bool Inline>
__global__ void iq4_linear(const Block* w,const float* x,float* y,int cols,int rows,int batch){
    int row=blockIdx.x,b=blockIdx.y,lane=threadIdx.x;if(row>=rows||b>=batch)return;
    float sum=0;int blocks=cols/256;const Block* wr=w+row*blocks;
    for(int i=lane;i<cols;i+=128)sum=fmaf(weight(wr[i/256],i%256,Inline),x[b*cols+i],sum);
    __shared__ float partial[128];partial[lane]=sum;__syncthreads();
    for(int n=64;n>0;n/=2){if(lane<n)partial[lane]+=partial[lane+n];__syncthreads();}
    if(lane==0)y[b*rows+row]=partial[0];
}
__device__ int bytes_per_256(int type){switch(type){case 0:return 1024;case 8:return 272;case 10:return 84;case 11:return 110;case 12:return 144;case 13:return 176;case 14:return 210;case 17:return 74;case 18:return 98;case 20:return 144;case 21:return 110;case 22:return 82;case 23:return 136;default:return 0;}}
__device__ void decode_256(const uint8_t* p,float* out,int type){switch(type){
case 0:for(int i=0;i<256;i++)out[i]=((const float*)p)[i];break;
case 8:dequantize_row_q8_0((const block_q8_0*)p,out,256);break;
case 10:dequantize_row_q2_K((const block_q2_K*)p,out,256);break;
case 11:dequantize_row_q3_K((const block_q3_K*)p,out,256);break;
case 12:dequantize_row_q4_K((const block_q4_K*)p,out,256);break;
case 13:dequantize_row_q5_K((const block_q5_K*)p,out,256);break;
case 14:dequantize_row_q6_K((const block_q6_K*)p,out,256);break;
case 17:dequantize_row_iq2_xs((const block_iq2_xs*)p,out,256);break;
case 18:dequantize_row_iq3_xxs((const block_iq3_xxs*)p,out,256);break;
case 20:dequantize_row_iq4_nl((const block_iq4_nl*)p,out,256);break;
case 21:dequantize_row_iq3_s((const block_iq3_s*)p,out,256);break;
case 22:dequantize_row_iq2_s((const block_iq2_s*)p,out,256);break;
case 23:dequantize_row_iq4_xs((const block_iq4_xs*)p,out,256);break;
}}
// Compatibility path: compressed weights, one tile decoded in LDS.
// Parallel element decoding for K formats, derived from pinned GGML formulas.
__device__ float element_k(const uint8_t* p,int j,int type){
    if(type==0)return ((const float*)p)[j];
    if(type==8){auto* b=((const block_q8_0*)p)+j/32;return ai_half(b->d)*b->qs[j%32];}
    if(type==10){auto* b=(const block_q2_K*)p;int chunk=j/128,sub=(j%128)/32,lane=j%32;int scale=b->scales[j/16];int value=(b->qs[chunk*32+lane]>>(2*sub))&3;return ai_half(b->data.d)*(scale&15)*value-ai_half(b->data.dmin)*(scale>>4);}
    if(type==11){auto* b=(const block_q3_K*)p;int group=j/16,chunk=j/128,sub=(j%128)/32,lane=j%32;int scale=((b->scales[group%8]>>(4*(group/8)))&15)|(((b->scales[8+group%4]>>(2*(group/4)))&3)<<4);int value=((b->qs[chunk*32+lane]>>(2*sub))&3)-(b->hmask[lane]&(1<<(chunk*4+sub))?0:4);return ai_half(b->d)*(scale-32)*value;}
    if(type==20){auto* b=((const block_iq4_nl*)p)+j/32;int i=j%32;int code=(b->qs[i%16]>>(i<16?0:4))&15;return ai_half(b->d)*kvalues_iq4nl[code];}
    if(type==21){auto* b=(const block_iq3_s*)p;int ib=j/32,l=(j%32)/8,z=j%8,half=z/4;int scale=(b->scales[ib/2]>>(4*(ib%2)))&15;int index=b->qs[ib*8+2*l+half]|(((b->qh[ib]>>(2*l+half))&1)<<8);auto* grid=(const uint8_t*)(iq3s_grid+index);return ai_half(b->d)*(1+2*scale)*grid[z%4]*(b->signs[ib*4+l]&(1<<z)?-1.f:1.f);}
    if(type==17){auto* b=(const block_iq2_xs*)p;int ib=j/32,l=(j%32)/8,z=j%8;int packed=b->qs[ib*4+l];int scale=(b->scales[ib]>>(4*(l/2)))&15;auto* grid=(const uint8_t*)(iq2xs_grid+(packed&511));int signs=ksigns_iq2xs[packed>>9];float d=ai_half(b->d)*(0.5f+scale)*0.25f;return d*grid[z]*(signs&(1<<z)?-1.f:1.f);}
    if(type==22){auto* b=(const block_iq2_s*)p;int ib=j/32,l=(j%32)/8,z=j%8;int index=b->qs[ib*4+l]|(((b->qh[ib]>>(2*l))&3)<<8);int scale=(b->scales[ib]>>(4*(l/2)))&15;auto* grid=(const uint8_t*)(iq2s_grid+index);int signs=b->qs[32+ib*4+l];float d=ai_half(b->d)*(0.5f+scale)*0.25f;return d*grid[z]*(signs&(1<<z)?-1.f:1.f);}
    if(type==18){auto* b=(const block_iq3_xxs*)p;int ib=j/32,l=(j%32)/8,z=j%8;const uint8_t* a=b->qs+64+4*ib;uint32_t aux=uint32_t(a[0])|(uint32_t(a[1])<<8)|(uint32_t(a[2])<<16)|(uint32_t(a[3])<<24);int signs=ksigns_iq2xs[(aux>>(7*l))&127];auto* grid=(const uint8_t*)(iq3xxs_grid+b->qs[ib*8+2*l+z/4]);float d=ai_half(b->d)*(0.5f+(aux>>28))*0.5f;return d*grid[z%4]*(signs&(1<<z)?-1.f:1.f);}
    if(type==12||type==13){
        int group=j/32,lane=j%32;uint8_t scale,minimum;
        if(type==12){auto* b=(const block_q4_K*)p;get_scale_min_k4(group,b->scales,&scale,&minimum);int value=(b->qs[(group/2)*32+lane]>>(4*(group%2)))&15;return ai_half(b->data.d)*scale*value-ai_half(b->data.dmin)*minimum;}
        auto* b=(const block_q5_K*)p;get_scale_min_k4(group,b->scales,&scale,&minimum);int value=((b->qs[(group/2)*32+lane]>>(4*(group%2)))&15)+(b->qh[lane]&(1<<group)?16:0);return ai_half(b->data.d)*scale*value-ai_half(b->data.dmin)*minimum;
    }
    auto* b=(const block_q6_K*)p;int chunk=j/128,sub=(j%128)/32,lane=j%32;
    int value=((b->ql[chunk*64+lane+(sub%2)*32]>>(4*(sub/2)))&15)|(((b->qh[chunk*32+lane]>>(2*sub))&3)<<4);
    return ai_half(b->d)*b->scales[chunk*8+sub*2+lane/16]*(value-32);
}
__global__ void parallel_k_linear(const uint8_t* w,const float* x,float* y,int cols,int rows,int batch,int type){
    int row=blockIdx.x,b=blockIdx.y,lane=threadIdx.x;if(row>=rows||b>=batch)return;
    __shared__ float partial[128];float sum=0;int stride=bytes_per_256(type);const uint8_t* wr=w+size_t(row)*(cols/256)*stride;
    for(int tile=0;tile<cols/256;tile++){const uint8_t* p=wr+tile*stride;sum=fmaf(element_k(p,lane,type),x[b*cols+tile*256+lane],sum);sum=fmaf(element_k(p,lane+128,type),x[b*cols+tile*256+lane+128],sum);}
    partial[lane]=sum;__syncthreads();for(int n=64;n>0;n/=2){if(lane<n)partial[lane]+=partial[lane+n];__syncthreads();}if(lane==0)y[b*rows+row]=partial[0];
}
// IQ3_S table entries pack four magnitude bytes in little-endian order.
// Opt-in experiment: aligned word load instead of byte-addressed table load.
// Preserves the original dequant multiplication order and FP32 value.
__device__ float iq3s_word_value(const uint8_t* p,int j){
 auto* b=(const block_iq3_s*)p;int ib=j/32,l=(j%32)/8,z=j%8,half=z/4;
 int scale=(b->scales[ib/2]>>(4*(ib%2)))&15;
 int index=b->qs[ib*8+2*l+half]|(((b->qh[ib]>>(2*l+half))&1)<<8);
 uint32_t word=iq3s_grid[index];float magnitude=float((word>>(8*(z%4)))&255u);
 return ai_half(b->d)*(1+2*scale)*magnitude*(b->signs[ib*4+l]&(1<<z)?-1.f:1.f);
}
// Serial decoder prioritizes correctness; it is not a throughput claim.
__global__ void quant_linear(const uint8_t* w,const float* x,float* y,int cols,int rows,int batch,int type){
    int row=blockIdx.x,b=blockIdx.y,lane=threadIdx.x;if(row>=rows||b>=batch)return;
    __shared__ float decoded[256],partial[128];float sum=0;int stride=bytes_per_256(type);
    const uint8_t* wr=w+size_t(row)*(cols/256)*stride;
    for(int tile=0;tile<cols/256;tile++){
        if(lane==0)decode_256(wr+tile*stride,decoded,type);__syncthreads();
        sum=fmaf(decoded[lane],x[b*cols+tile*256+lane],sum);
        sum=fmaf(decoded[lane+128],x[b*cols+tile*256+lane+128],sum);__syncthreads();
    }
    partial[lane]=sum;__syncthreads();for(int n=64;n>0;n/=2){if(lane<n)partial[lane]+=partial[lane+n];__syncthreads();}
    if(lane==0)y[b*rows+row]=partial[0];
}
__global__ void quant_debug(const uint8_t* w,float* y,int type){if(threadIdx.x==0){y[0]=type;y[1]=bytes_per_256(type);y[2]=ai_half(((const block_q5_K*)w)->data.d);y[3]=ai_half(((const block_q5_K*)w)->data.dmin);decode_256(w,y+4,type);}}
__global__ void quant_dequant(const uint8_t* w,float* y,int type){if(threadIdx.x==0)decode_256(w+blockIdx.x*bytes_per_256(type),y+blockIdx.x*256,type);}
extern "C" {
__global__ void rms_kernel(const float* x,const float* w,float* out,int cols){
    __shared__ float partial[256];int lane=threadIdx.x;float s=0;
    for(int i=lane;i<cols;i+=256)s+=x[i]*x[i];partial[lane]=s;__syncthreads();
    for(int d=128;d;d/=2){if(lane<d)partial[lane]+=partial[lane+d];__syncthreads();}
    float inv=rsqrtf(partial[0]/cols+1e-6f);
    for(int i=lane;i<cols;i+=256)out[i]=x[i]*inv*w[i];
}
__global__ void attention_kernel(const float* q,const float* k,const float* v,const float* gate,float* out,int count){
    __shared__ float scores[512],denom;int h=blockIdx.x,lane=threadIdx.x,kh=h/6;
    for(int t=lane;t<count;t+=256){float dot=0;for(int j=0;j<256;j++)dot+=q[h*256+j]*k[t*1024+kh*256+j];scores[t]=dot/16.f;}
    __syncthreads();if(lane==0){float m=-INFINITY;for(int t=0;t<count;t++)m=fmaxf(m,scores[t]);float sum=0;for(int t=0;t<count;t++){scores[t]=expf(scores[t]-m);sum+=scores[t];}denom=sum;}__syncthreads();
    float sum=0;for(int t=0;t<count;t++)sum+=scores[t]/denom*v[t*1024+kh*256+lane];
    float g=gate[h*256+lane];float sigmoid=g>=0?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));out[h*256+lane]=sum*sigmoid;
}
// Diagnostic precision variant: same double sum/sqrt and rounded FP32 inverse
// as the scalar oracle; keep the frozen FP32 reduction selectable.
__global__ void rms_stable_kernel(const float* x,const float* w,float* out,int cols){
    __shared__ double partial[256];int lane=threadIdx.x;double s=0;
    for(int i=lane;i<cols;i+=256){double v=x[i];s+=v*v;}partial[lane]=s;__syncthreads();
    for(int d=128;d;d/=2){if(lane<d)partial[lane]+=partial[lane+d];__syncthreads();}
    float inv=float(1.0/sqrt(partial[0]/double(cols)+double(1e-6f)));
    for(int i=lane;i<cols;i+=256)out[i]=x[i]*inv*w[i];
}
int ai_rms(const float* x,const float* w,float* out,int cols){if(cols<1)return hipErrorInvalidValue;const char* stable=std::getenv("AMD_INFER_STABLE_RMS");if(stable&&stable[0]=='1'){hipLaunchKernelGGL(rms_stable_kernel,dim3(1),dim3(256),0,ai_stream(),x,w,out,cols);}else{hipLaunchKernelGGL(rms_kernel,dim3(1),dim3(256),0,ai_stream(),x,w,out,cols);}return hipGetLastError();}
__global__ void residual_kernel(float* hidden,const float* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<5120)hidden[i]=__fadd_rn(hidden[i],output[i]);}
int ai_residual(float* hidden,const float* output){hipLaunchKernelGGL(residual_kernel,dim3(20),dim3(256),0,ai_stream(),hidden,output);return hipGetLastError();}
// Same ordered FP64 norm and split FP32 RoPE arithmetic as the host oracle.
// RoPE coefficients are computed once per position by Rust to preserve libm policy.
__global__ void attention_prepare_kernel(const float* qg,const float* k,const float* w,const float* rotation,float* query,float* key){
 #pragma clang fp contract(off)
 int h=blockIdx.x,j=threadIdx.x;bool isq=h<24;int kh=h-24;
 const float* x=isq?qg+h*512:k+kh*256;const float* weight=w+(isq?0:256);
 __shared__ float normalized[256],inv;
 if(j==0){double sum=0;for(int n=0;n<256;n++){double v=x[n];sum+=v*v;}inv=float(1.0/sqrt(sum/256.0+double(1e-6f)));}
 __syncthreads();normalized[j]=(x[j]*inv)*weight[j];__syncthreads();
 float value=normalized[j];if(j<32)value=normalized[j]*rotation[j]-normalized[j+32]*rotation[j+32];
 else if(j<64)value=normalized[j-32]*rotation[j]+normalized[j]*rotation[j-32];
 if(isq){query[h*256+j]=value;query[6144+h*256+j]=x[256+j];}else key[kh*256+j]=value;
}
int ai_attention_prepare(const float* qg,const float* k,const float* w,const float* rotation,float* query,float* key){hipLaunchKernelGGL(attention_prepare_kernel,dim3(28),dim3(256),0,ai_stream(),qg,k,w,rotation,query,key);return hipGetLastError();}
__global__ void attention_scalar_kernel(const float* q,const float* k,const float* v,const float* gate,float* out,int count){
    #pragma clang fp contract(off)
    __shared__ float scores[512],denom;int h=blockIdx.x,lane=threadIdx.x,kh=h/6;
    for(int t=lane;t<count;t+=256){float dot=0;for(int j=0;j<256;j++)dot+=q[h*256+j]*k[t*1024+kh*256+j];scores[t]=dot/16.f;}
    __syncthreads();if(lane==0){float m=-INFINITY;for(int t=0;t<count;t++)m=fmaxf(m,scores[t]);float sum=0;for(int t=0;t<count;t++){scores[t]=float(exp(double(__fadd_rn(scores[t],-m))));sum=__fadd_rn(sum,scores[t]);}denom=sum;}__syncthreads();
    float sum=0;for(int t=0;t<count;t++){float probability=float(double(scores[t])/double(denom));sum+=probability*v[t*1024+kh*256+lane];}
    float g=gate[h*256+lane],e=float(exp(double(g>=0?-g:g))),d=__fadd_rn(1.f,e);float sigmoid=float(double(g>=0?1.f:e)/double(d));out[h*256+lane]=sum*sigmoid;
}
int ai_attention(const float* q,const float* k,const float* v,const float* gate,float* out,int count){if(count<1||count>512)return hipErrorInvalidValue;const char* flag=std::getenv("AMD_INFER_SCALAR_ATTENTION");if(flag&&flag[0]=='1'){hipLaunchKernelGGL(attention_scalar_kernel,dim3(24),dim3(256),0,ai_stream(),q,k,v,gate,out,count);}else{hipLaunchKernelGGL(attention_kernel,dim3(24),dim3(256),0,ai_stream(),q,k,v,gate,out,count);}return hipGetLastError();}
__global__ void swiglu_kernel(const float* gate,const float* up,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float g=gate[i];float sigmoid=g>=0?1.f/(1.f+expf(-g)):expf(g)/(1.f+expf(g));out[i]=g*sigmoid*up[i];}}
int ai_delta(float* state,const float* input,float* out);
int ai_dequant(const void* w,float* y,int cols,int type){if(cols<=0||cols%256)return hipErrorInvalidValue;hipLaunchKernelGGL(quant_dequant,dim3(cols/256),dim3(32),0,ai_stream(),(const uint8_t*)w,y,type);return hipGetLastError();}
int ai_debug_quant(const void* w,float* y,int type){hipLaunchKernelGGL(quant_debug,dim3(1),dim3(32),0,ai_stream(),(const uint8_t*)w,y,type);return hipGetLastError();}
int ai_alloc(void** p,size_t n){auto& owned=ai_execution();if(owned.status)return owned.status;int e=hipMalloc(p,n);if(!e)owned.allocations[ai_pointer(*p)]=n;return e;}
int ai_free(void* p){auto& owned=ai_execution();if(owned.stream){int e=hipStreamSynchronize(owned.stream);if(e)return e;}owned.invalidate(p);return hipFree(p);}
int ai_zero(void* p,size_t n){if(!ai_stream())return hipMemset(p,0,n);int e=hipMemsetAsync(p,0,n,ai_stream());return e?e:hipStreamSynchronize(ai_stream());}
int ai_h2d(void* dst,const void* src,size_t n){if(!ai_stream())return hipMemcpy(dst,src,n,hipMemcpyHostToDevice);int e=hipMemcpyAsync(dst,src,n,hipMemcpyHostToDevice,ai_stream());return e?e:hipStreamSynchronize(ai_stream());}
int ai_d2d(void* dst,const void* src,size_t n){return hipMemcpyAsync(dst,src,n,hipMemcpyDeviceToDevice,ai_stream());}
int ai_d2h(void* dst,const void* src,size_t n){if(!ai_stream())return hipMemcpy(dst,src,n,hipMemcpyDeviceToHost);int e=hipMemcpyAsync(dst,src,n,hipMemcpyDeviceToHost,ai_stream());return e?e:hipStreamSynchronize(ai_stream());}
const char* ai_error(int e){return hipGetErrorString((hipError_t)e);}
int ai_linear(const void* w,const float* x,float* y,int cols,int rows,int batch){
    if(cols<=0||cols%256||rows<=0||batch<1||batch>8)return hipErrorInvalidValue;
    const char* flag=std::getenv("AMD_INFER_INLINE_IQ4_LEVELS");
    if(flag&&flag[0]=='1'){hipLaunchKernelGGL((iq4_linear<true>),dim3(rows,batch),dim3(128),0,ai_stream(),(const Block*)w,x,y,cols,rows,batch);}
    else{hipLaunchKernelGGL((iq4_linear<false>),dim3(rows,batch),dim3(128),0,ai_stream(),(const Block*)w,x,y,cols,rows,batch);}
    return hipGetLastError();
}

// One block per value head, one lane per value dimension. Fuses state decay,
// prediction, rank-one update and query projection for B=1 Qwen DeltaNet.
__global__ void delta_step(float* state,const float* input,float* out){
    int h=blockIdx.x,j=threadIdx.x,kh=h%16;
    const float *q=input+kh*128,*k=input+2048+kh*128,*v=input+4096;
    float g=input[10240+h],beta=input[10288+h];
    float* row=state+(h*128+j)*128;float predicted=0,decay=expf(g);
    for(int i=0;i<128;i++){row[i]*=decay;predicted+=row[i]*k[i];}
    float delta=(v[h*128+j]-predicted)*beta,result=0;
    for(int i=0;i<128;i++){row[i]+=delta*k[i];result+=row[i]*q[i];}
    out[h*128+j]=result*0.08838834764831845f;
}
extern "C" int ai_delta(float* state,const float* input,float* out){
    hipLaunchKernelGGL(delta_step,dim3(48),dim3(128),0,ai_stream(),state,input,out);
    return hipGetLastError();
}
} // C ABI ends before templated GDN kernels.
__device__ float stable_sigmoid(float x){if(x>=0)return 1.f/(1.f+expf(-x));float e=expf(x);return e/(1.f+e);}
__device__ float scalar_sigmoid(float x){float e=float(exp(double(x>=0?-x:x)));float denom=__fadd_rn(1.f,e);return float(double(x>=0?1.f:e)/double(denom));}
__device__ float scalar_conv_dot(const float* h,const float* w,float x){
 #pragma clang fp contract(off)
 float result=h[0]*w[0]+h[1]*w[1];result+=h[2]*w[2];return result+x*w[3];
}
template<bool Scalar>
__global__ void gdn_conv_prepare(float* history,const float* raw,const float* weights,float* mixed){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<10240){float* h=history+c*3;const float* w=weights+c*4;float y;
 if(Scalar){y=scalar_conv_dot(h,w,raw[c]);}else{y=h[0]*w[0]+h[1]*w[1]+h[2]*w[2]+raw[c]*w[3];}
 mixed[c]=y*(Scalar?scalar_sigmoid(y):stable_sigmoid(y));h[0]=h[1];h[1]=h[2];h[2]=raw[c];}
 if(c<48){float x=raw[16384+c]+weights[40960+128+48+c];float s;if(Scalar){float e=float(exp(-double(fabsf(x))));float log=float(log1p(double(e)));s=__fadd_rn(fmaxf(x,0.f),log);}else{s=fmaxf(x,0.f)+log1pf(expf(-fabsf(x)));}mixed[10240+c]=s*weights[40960+128+c];mixed[10288+c]=Scalar?scalar_sigmoid(raw[16432+c]):stable_sigmoid(raw[16432+c]);}
}
template<bool Scalar>
__global__ void gdn_l2(float* mixed){
 int h=blockIdx.x,j=threadIdx.x;__shared__ double sum[128];float* x=mixed+h*128;double value=x[j];sum[j]=value*value;__syncthreads();for(int n=64;n>0;n/=2){if(j<n)sum[j]+=sum[j+n];__syncthreads();}float norm=float(sqrt(sum[0]+double(1e-6f)));x[j]=Scalar?float(double(x[j])/double(norm)):x[j]/norm;
}
template<bool Scalar>
__global__ void gdn_norm_gate(const float* core,const float* raw,const float* weights,float* gated){
 int h=blockIdx.x,j=threadIdx.x;__shared__ double sum[128];double value=core[h*128+j];sum[j]=value*value;__syncthreads();for(int n=64;n>0;n/=2){if(j<n)sum[j]+=sum[j+n];__syncthreads();}float inv=float(1.0/sqrt(sum[0]/128.+double(1e-6f)));float z=raw[10240+h*128+j];gated[h*128+j]=core[h*128+j]*inv*weights[40960+j]*z*(Scalar?scalar_sigmoid(z):stable_sigmoid(z));
}
// Bounded head task loop; all 128 threads follow the same head loop.
// No inter-block communication, token loop or global barrier. Column changes
// only physical storage; each value lane retains the original key-loop order.
template<bool Scalar, bool Column=false, bool Fuse=true>
__global__ void gdn_head_task(float* state,const float* input,float* core,const float* raw,const float* weights,float* gated){
 __shared__ double partial[128];int j=threadIdx.x;
 for(int h=blockIdx.x;h<48;h+=gridDim.x){int kh=h%16;
  const float* q=input+kh*128;const float* k=input+2048+kh*128;float decay=expf(input[10240+h]),predicted=0;
  for(int i=0;i<128;i++){int ix=h*16384+(Column?i*128+j:j*128+i);state[ix]*=decay;predicted+=state[ix]*k[i];}
  float delta=(input[4096+h*128+j]-predicted)*input[10288+h],result=0;
  for(int i=0;i<128;i++){int ix=h*16384+(Column?i*128+j:j*128+i);state[ix]+=delta*k[i];result+=state[ix]*q[i];}
  float value=result*0.08838834764831845f;core[h*128+j]=value;
  if constexpr(Fuse){double v=value;partial[j]=v*v;__syncthreads();
   for(int n=64;n>0;n/=2){if(j<n)partial[j]+=partial[j+n];__syncthreads();}
   float inv=float(1.0/sqrt(partial[0]/128.+double(1e-6f))),z=raw[10240+h*128+j];
   gated[h*128+j]=value*inv*weights[40960+j]*z*(Scalar?scalar_sigmoid(z):stable_sigmoid(z));
   if(h+gridDim.x<48)__syncthreads();
  }
 }
}
extern "C" {
int ai_gdn_pipeline_layout(float* state,float* history,const float* weights,const float* raw,float* mixed,float* core,float* gated,int column){
 if(column!=0&&column!=1)return hipErrorInvalidValue;
 const char* flag=std::getenv("AMD_INFER_SCALAR_GDN");bool scalar=flag&&flag[0]=='1';
 if(scalar){hipLaunchKernelGGL((gdn_conv_prepare<true>),dim3(40),dim3(256),0,ai_stream(),history,raw,weights,mixed);}else{hipLaunchKernelGGL((gdn_conv_prepare<false>),dim3(40),dim3(256),0,ai_stream(),history,raw,weights,mixed);}int e=hipGetLastError();if(e)return e;e=ai_gdn_mark(5);if(e)return e;
 if(scalar){hipLaunchKernelGGL((gdn_l2<true>),dim3(32),dim3(128),0,ai_stream(),mixed);}else{hipLaunchKernelGGL((gdn_l2<false>),dim3(32),dim3(128),0,ai_stream(),mixed);}e=hipGetLastError();if(e)return e;e=ai_gdn_mark(6);if(e)return e;
 const char* fused=std::getenv("AMD_INFER_GDN_HEAD_FUSE");
 if(fused&&fused[0]=='1'){
  const char* value=std::getenv("AMD_INFER_GDN_HEAD_BLOCKS");int blocks=value?std::atoi(value):48;
  if(blocks!=24&&blocks!=32&&blocks!=48)return hipErrorInvalidValue;
  if(column){
   if(scalar){hipLaunchKernelGGL((gdn_head_task<true,true>),dim3(blocks),dim3(128),0,ai_stream(),state,mixed,core,raw,weights,gated);}
   else{hipLaunchKernelGGL((gdn_head_task<false,true>),dim3(blocks),dim3(128),0,ai_stream(),state,mixed,core,raw,weights,gated);}
  }else{
   if(scalar){hipLaunchKernelGGL((gdn_head_task<true>),dim3(blocks),dim3(128),0,ai_stream(),state,mixed,core,raw,weights,gated);}
   else{hipLaunchKernelGGL((gdn_head_task<false>),dim3(blocks),dim3(128),0,ai_stream(),state,mixed,core,raw,weights,gated);}
  }
  e=hipGetLastError();if(e)return e;e=ai_gdn_mark(7);if(e)return e;return ai_gdn_mark(8);
 }
 if(column){
  hipLaunchKernelGGL((gdn_head_task<true,true,false>),dim3(48),dim3(128),0,ai_stream(),state,mixed,core,raw,weights,gated);
  e=hipGetLastError();
 }else{e=ai_delta(state,mixed,core);}if(e)return e;e=ai_gdn_mark(7);if(e)return e;
 if(scalar){hipLaunchKernelGGL((gdn_norm_gate<true>),dim3(48),dim3(128),0,ai_stream(),core,raw,weights,gated);}else{hipLaunchKernelGGL((gdn_norm_gate<false>),dim3(48),dim3(128),0,ai_stream(),core,raw,weights,gated);}e=hipGetLastError();if(e)return e;return ai_gdn_mark(8);
}
int ai_gdn_pipeline(float* state,float* history,const float* weights,const float* raw,float* mixed,float* core,float* gated){return ai_gdn_pipeline_layout(state,history,weights,raw,mixed,core,gated,0);}
int ai_sync(){return ai_stream()?hipStreamSynchronize(ai_stream()):hipDeviceSynchronize();}
int ai_memory(size_t* free_bytes,size_t* total_bytes){return hipMemGetInfo(free_bytes,total_bytes);}
} // C ABI: templated format specialization below.
// Phase11 exact-policy candidates:1/2/6 rows8/16/2,3 byte-permute IQ4,
//4 block-local IQ3 codebook,5 aligned uint2 packets,7 one-tile packet prefetch.
inline int ai_tune_mode(){const char* p=std::getenv("AMD_INFER_GEMV_TUNE");int m=p?std::atoi(p):0;return m>=0&&m<=7?m:0;}
constexpr int ai_tune_rows(int m){return m==1?8:m==2?16:m==6?2:4;}
__device__ float iq4_permute_level(int code){
 uint32_t a=code<8?0xbfad9881u:0x26190d01u,b=code<8?0xf6eaddcfu:0x71594535u;
 int v=int(__byte_perm(a,b,unsigned(code&7))&255u);return float(v>=128?v-256:v);
}
template<int Mode> __device__ uint32_t iq4_packet(const uint8_t* p,int lane){
 if constexpr(Mode==5){uint2 q=((const uint2*)(p+8))[lane/2];return lane&1?q.y:q.x;}
 else return ((const uint32_t*)(p+8))[lane];
}
// Phase10:64 IQ3_S codebook words per tile are owned by32 lanes.
// Each lane gathers two words, then shuffles distribute their four bytes.
// Keep the original dequant scale, magnitude and sign multiplication order.
template<bool Shared=false> __device__ uint32_t cooperative_iq3_word(const uint8_t* p,int lane,int half,const uint32_t* lut=nullptr){
 auto* b=(const block_iq3_s*)p;int j=4*lane+128*half,ib=j/32,l=(j%32)/8,h=(j%8)/4;
 int index=b->qs[ib*8+2*l+h]|(((b->qh[ib]>>(2*l+h))&1)<<8);
 if constexpr(Shared)return lut[index];else return iq3s_grid[index];
}
__device__ float cooperative_iq3_value(const uint8_t* p,uint32_t packet,int lane,int r,int half){
 auto* b=(const block_iq3_s*)p;int j=lane+32*r+128*half,ib=j/32,l=(j%32)/8,z=j%8;
 uint32_t word=__shfl(packet,(lane+32*r)/4,32);float magnitude=float((word>>(8*(j%4)))&255u);
 int scale=(b->scales[ib/2]>>(4*(ib%2)))&15;
 return ai_half(b->d)*(1+2*scale)*magnitude*(b->signs[ib*4+l]&(1<<z)?-1.f:1.f);
}
// Phase9: one aligned packet word per lane loads all128 IQ4_XS bytes.
// Wave shuffles reuse the packet for both nibble halves. This reduces redundant
// load instructions, not the unique compressed matrix payload. No global barrier.
template<bool CachedLevels,int Mode=0>
__device__ float cooperative_iq4_value(const Block& b,uint32_t packet,float cached_level,int lane,int r,int half){
 int s=r+4*half;
 uint32_t word=__shfl(packet,4*s+((lane&15)>>2),32);
 int code=(word>>(8*(lane&3)+4*(lane>>4)))&15;
 int sc=((b.scales_l[s/2]>>(4*(s%2)))&15)|(((b.scales_h>>(2*s))&3)<<4);
 return ai_half(b.d)*(sc-32)*(Mode==3?iq4_permute_level(code):(CachedLevels?__shfl(cached_level,code,32):levels[code]));
}
template<int Kind,bool Cooperative=false,int Mode=0>
__global__ void warp_quant_linear(const uint8_t* w,const float* x,float* y,int cols,int rows,int batch){
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 #if defined(__HIP_DEVICE_COMPILE__)
 warp=__builtin_amdgcn_readfirstlane(warp);
 #endif
 __shared__ uint32_t lut[Mode==4&&Kind==21?512:1];
 if constexpr(Mode==4&&Kind==21){for(int i=threadIdx.x;i<512;i+=blockDim.x)lut[i]=iq3s_grid[i];__syncthreads();}
 int row=blockIdx.x*ai_tune_rows(Mode)+warp,b=blockIdx.y;
 if(row>=rows||b>=batch)return;
 const uint8_t* wr=w+size_t(row)*(cols/256)*bytes_per_256(Kind);float sums[4]={0,0,0,0};
 float cached_level=0;if constexpr(Cooperative&&Kind==23)cached_level=levels[lane&15];
 uint32_t prefetched=0;if constexpr(Cooperative&&Kind==23&&Mode==7)prefetched=iq4_packet<0>(wr,lane);
 for(int tile=0;tile<cols/256;tile++){const uint8_t* p=wr+tile*bytes_per_256(Kind);
  uint32_t packet=0,iq3[2]={0,0};
  if constexpr(Cooperative&&Kind==23){if constexpr(Mode==7){packet=prefetched;if(tile+1<cols/256)prefetched=iq4_packet<0>(p+bytes_per_256(Kind),lane);}else packet=iq4_packet<Mode>(p,lane);}
  if constexpr(Cooperative&&Kind==21){iq3[0]=cooperative_iq3_word<Mode==4>(p,lane,0,lut);iq3[1]=cooperative_iq3_word<Mode==4>(p,lane,1,lut);}
  #pragma unroll
  for(int r=0;r<4;r++){int j=lane+32*r;
   float a=Cooperative&&Kind==21?cooperative_iq3_value(p,iq3[0],lane,r,0):(Kind==23?(Cooperative?cooperative_iq4_value<true,Mode>(*(const Block*)p,packet,cached_level,lane,r,0):weight(*(const Block*)p,j,false)):element_k(p,j,Kind));
   float c=Cooperative&&Kind==21?cooperative_iq3_value(p,iq3[1],lane,r,1):(Kind==23?(Cooperative?cooperative_iq4_value<true,Mode>(*(const Block*)p,packet,cached_level,lane,r,1):weight(*(const Block*)p,j+128,false)):element_k(p,j+128,Kind));
   sums[r]=fmaf(a,x[b*cols+tile*256+j],sums[r]);
   sums[r]=fmaf(c,x[b*cols+tile*256+j+128],sums[r]);
  }
 }
 float sum=(sums[0]+sums[2])+(sums[1]+sums[3]);
 for(int n=16;n>0;n/=2)sum+=__shfl_down(sum,n,32);
 if(lane==0)y[b*rows+row]=sum;
}
template<int Kind> const void* ai_tuned_linear_pointer(int mode){
 switch(mode){
 #define AI_TUNED_PTR(M) case M:return (const void*)warp_quant_linear<Kind,true,M>;
 AI_TUNED_PTR(1) AI_TUNED_PTR(2) AI_TUNED_PTR(3) AI_TUNED_PTR(4) AI_TUNED_PTR(5) AI_TUNED_PTR(6) AI_TUNED_PTR(7)
 #undef AI_TUNED_PTR
 default:return (const void*)warp_quant_linear<Kind,true>;
 }
}
template<int Kind> int ai_launch_tuned_linear(int mode,const void* w,const float* x,float* y,int cols,int rows,int batch){
 switch(mode){
 #define AI_TUNED_LAUNCH(M) case M:hipLaunchKernelGGL((warp_quant_linear<Kind,true,M>),dim3((rows+ai_tune_rows(M)-1)/ai_tune_rows(M),batch),dim3(32*ai_tune_rows(M)),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows,batch);break;
 AI_TUNED_LAUNCH(1) AI_TUNED_LAUNCH(2) AI_TUNED_LAUNCH(3) AI_TUNED_LAUNCH(4) AI_TUNED_LAUNCH(5) AI_TUNED_LAUNCH(6) AI_TUNED_LAUNCH(7)
 #undef AI_TUNED_LAUNCH
 default:return hipErrorInvalidValue;
 }return hipGetLastError();
}
inline bool ai_regroup_requested(){const char* p=std::getenv("AMD_INFER_FP32_REGROUP");return p&&p[0]=='1';}
int ai_launch_regrouped_model(const void*,const float*,float*,int,int,int);
template<int Kind> __global__ void regrouped_probe(const uint8_t*,const float*,float*,int,int);
template<int G,int U> void ai_launch_regrouped_gate(const void*,const void*,float*,float*,int);
const void* ai_regrouped_gate_pointer(int,int);
extern "C" {
int ai_quant_linear(const void* w,const float* x,float* y,int cols,int rows,int batch,int type){
    if(cols<=0||cols%256||rows<=0||batch<1||batch>8)return hipErrorInvalidValue;
    switch(type){case 0:case 8:case 10:case 11:case 12:case 13:case 14:case 17:case 18:case 20:case 21:case 22:case 23:break;default:return hipErrorInvalidValue;}
    const char* warp=std::getenv("AMD_INFER_WARP_GEMV");
    if(warp&&warp[0]=='1'&&rows>=512){
      if(ai_regroup_requested()&&batch==1&&(type==21||type==23))return ai_launch_regrouped_model(w,x,y,cols,rows,type);

      const char* all=std::getenv("AMD_INFER_COOP_QUANT_GEMV");
      if(all&&all[0]=='1'){int mode=ai_tune_mode();if(mode){
       if(type==23)return ai_launch_tuned_linear<23>(mode,w,x,y,cols,rows,batch);
       if(type==21)return ai_launch_tuned_linear<21>(mode,w,x,y,cols,rows,batch);
      }
       if(type==23){hipLaunchKernelGGL((warp_quant_linear<23,true>),dim3((rows+3)/4,batch),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows,batch);return hipGetLastError();}
       if(type==21){hipLaunchKernelGGL((warp_quant_linear<21,true>),dim3((rows+3)/4,batch),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows,batch);return hipGetLastError();}
      }
      #define AI_WARP(T) case T:hipLaunchKernelGGL((warp_quant_linear<T>),dim3((rows+3)/4,batch),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows,batch);break;
      switch(type){AI_WARP(0) AI_WARP(8) AI_WARP(10) AI_WARP(11) AI_WARP(12) AI_WARP(13) AI_WARP(14) AI_WARP(17) AI_WARP(18) AI_WARP(20) AI_WARP(21) AI_WARP(22) AI_WARP(23)}
      #undef AI_WARP
      return hipGetLastError();
    }
    if(type==23)return ai_linear(w,x,y,cols,rows,batch);
    hipLaunchKernelGGL(parallel_k_linear,dim3(rows,batch),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows,batch,type);return hipGetLastError();
}
// Same-stream dependency chain: intermediates stay on device. No host barrier
// between RMS, the two projections, SwiGLU and the down projection.
int ai_gdn_projected_raw_layout(const void* const* matrices,const int* types,const float* x,float* raw,float* state,float* history,const float* weights,float* mixed,float* core,float* gated,float* output,int column){
 int e=ai_gdn_mark(0);if(e)return e;e=ai_gdn_mark(1);if(e)return e;e=ai_quant_linear(matrices[0],x,raw,5120,10240,1,types[0]);if(e)return e;e=ai_gdn_mark(1);if(e)return e;
 e=ai_quant_linear(matrices[1],x,raw+10240,5120,6144,1,types[1]);if(e)return e;e=ai_gdn_mark(2);if(e)return e;
 e=ai_quant_linear(matrices[2],x,raw+16384,5120,48,1,types[2]);if(e)return e;e=ai_gdn_mark(3);if(e)return e;
 e=ai_quant_linear(matrices[3],x,raw+16432,5120,48,1,types[3]);if(e)return e;e=ai_gdn_mark(4);if(e)return e;
 e=ai_gdn_pipeline_layout(state,history,weights,raw,mixed,core,gated,column);if(e)return e;
 e=ai_quant_linear(matrices[4],gated,output,6144,5120,1,types[4]);if(e)return e;e=ai_gdn_mark(9);if(e)return e;return ai_gdn_finish();
}
int ai_gdn_projected_layout(const void* const* matrices,const int* types,const float* x,float* raw,float* state,float* history,const float* weights,float* mixed,float* core,float* gated,float* output,int column){
 if(column!=0&&column!=1)return hipErrorInvalidValue;
 auto body=[&](){return ai_gdn_projected_raw_layout(matrices,types,x,raw,state,history,weights,mixed,core,gated,output,column);};
 if(!ai_graph_requested()||ai_gdn_profile().enabled)return body();
 std::vector<uintptr_t> key{1,ai_math_key(),uintptr_t(column),ai_pointer(x),ai_pointer(raw),ai_pointer(state),ai_pointer(history),ai_pointer(weights),ai_pointer(mixed),ai_pointer(core),ai_pointer(gated),ai_pointer(output)};
 for(int i=0;i<5;i++){key.push_back(ai_pointer(matrices[i]));key.push_back(types[i]);}
 return ai_graph_dispatch(key,body);
}
int ai_gdn_projected(const void* const* matrices,const int* types,const float* x,float* raw,float* state,float* history,const float* weights,float* mixed,float* core,float* gated,float* output){return ai_gdn_projected_layout(matrices,types,x,raw,state,history,weights,mixed,core,gated,output,0);}
// Persistent row tasks only: no cross-block barrier. Both projections consume
// the same normalized input; activation is disjoint from final down output.
} // extern C: templates below require C++ linkage.
template<int GT,int UT,bool Inline=false>
__global__ void persistent_gate_up(const uint8_t* gate,const uint8_t* up,const float* x,float* act,int gt_arg,int ut_arg,int warp_reduce){
 const int gt=GT<0?gt_arg:GT,ut=UT<0?ut_arg:UT;
 int lane=threadIdx.x;__shared__ float pg[128],pu[128];
 for(int row=blockIdx.x;row<17408;row+=gridDim.x){
  const uint8_t* gr=gate+size_t(row)*20*bytes_per_256(gt);const uint8_t* ur=up+size_t(row)*20*bytes_per_256(ut);
  float gs=0,us=0;
  for(int tile=0;tile<20;tile++){const uint8_t* gp=gr+tile*bytes_per_256(gt);const uint8_t* uptr=ur+tile*bytes_per_256(ut);
   for(int j=lane;j<256;j+=128){float xv=x[tile*256+j];float g=gt==23?weight(*(const Block*)gp,j,Inline):element_k(gp,j,gt);float u=ut==23?weight(*(const Block*)uptr,j,Inline):element_k(uptr,j,ut);gs=fmaf(g,xv,gs);us=fmaf(u,xv,us);}
  }
  if(warp_reduce){
   for(int n=16;n>0;n/=2){gs+=__shfl_down(gs,n,32);us+=__shfl_down(us,n,32);}
   if(lane%32==0){pg[lane/32]=gs;pu[lane/32]=us;}__syncthreads();
   if(lane<32){gs=lane<4?pg[lane]:0;us=lane<4?pu[lane]:0;for(int n=16;n>0;n/=2){gs+=__shfl_down(gs,n,32);us+=__shfl_down(us,n,32);}if(lane==0)act[row]=(gs/(1.f+expf(-gs)))*us;}
  }else{
   pg[lane]=gs;pu[lane]=us;__syncthreads();for(int n=64;n>0;n/=2){if(lane<n){pg[lane]+=pg[lane+n];pu[lane]+=pu[lane+n];}__syncthreads();}
   if(lane==0)act[row]=(pg[0]/(1.f+expf(-pg[0])))*pu[0];
  }
  __syncthreads();
 }
}
// Four independent output rows per block. Each warp streams one GGUF row;
// there is no block-wide barrier or cross-warp reduction in this variant.
template<int GT,int UT,bool Inline=false>
__global__ void warp_row_gate_up(const uint8_t* gate,const uint8_t* up,const float* x,float* act,int gt_arg,int ut_arg){
 const int gt=GT<0?gt_arg:GT,ut=UT<0?ut_arg:UT;int lane=threadIdx.x%32,warp=threadIdx.x/32;
 for(int row=blockIdx.x*4+warp;row<17408;row+=gridDim.x*4){
  const uint8_t* gr=gate+size_t(row)*20*bytes_per_256(gt);const uint8_t* ur=up+size_t(row)*20*bytes_per_256(ut);float gs=0,us=0;
  for(int tile=0;tile<20;tile++){const uint8_t* gp=gr+tile*bytes_per_256(gt);const uint8_t* uptr=ur+tile*bytes_per_256(ut);for(int j=lane;j<256;j+=32){float xv=x[tile*256+j];float g=gt==23?weight(*(const Block*)gp,j,Inline):element_k(gp,j,gt);float u=ut==23?weight(*(const Block*)uptr,j,Inline):element_k(uptr,j,ut);gs=fmaf(g,xv,gs);us=fmaf(u,xv,us);}}
  for(int n=16;n>0;n/=2){gs+=__shfl_down(gs,n,32);us+=__shfl_down(us,n,32);}
  if(lane==0)act[row]=(gs/(1.f+expf(-gs)))*us;
 }
}
// Four independent accumulators per lane reconstruct the reference 128-lane
// GEMV tree, while retaining four warp-owned output rows per block.
template<int GT,int UT,bool Inline=false,bool PackedGrid=false,bool UniformRow=false,bool Cooperative=false,bool CachedLevels=false,bool CooperativeIQ3=false,int Mode=0>
__global__ void scalar_warp_gate_up(const uint8_t* gate,const uint8_t* up,const float* x,float* act,int gt_arg,int ut_arg){
 const int gt=GT<0?gt_arg:GT,ut=UT<0?ut_arg:UT;int lane=threadIdx.x%32,warp=threadIdx.x/32;
 #if defined(__HIP_DEVICE_COMPILE__)
 if constexpr(UniformRow)warp=__builtin_amdgcn_readfirstlane(warp);
 #endif

 __shared__ uint32_t lut[Mode==4&&(GT==21||UT==21)?512:1];
 if constexpr(Mode==4&&(GT==21||UT==21)){for(int i=threadIdx.x;i<512;i+=blockDim.x)lut[i]=iq3s_grid[i];__syncthreads();}
 float cached_level=0;
 if constexpr(Cooperative&&CachedLevels)cached_level=levels[lane&15];
 for(int row=blockIdx.x*ai_tune_rows(Mode)+warp;row<17408;row+=gridDim.x*ai_tune_rows(Mode)){
  const uint8_t* gr=gate+size_t(row)*20*bytes_per_256(gt);const uint8_t* ur=up+size_t(row)*20*bytes_per_256(ut);float gs[4]={0,0,0,0},us[4]={0,0,0,0};
  uint32_t pre_gate=0,pre_up=0;
  if constexpr(Mode==7&&Cooperative&&GT==23)pre_gate=iq4_packet<0>(gr,lane);
  if constexpr(Mode==7&&Cooperative&&UT==23)pre_up=iq4_packet<0>(ur,lane);
  for(int tile=0;tile<20;tile++){const uint8_t* gp=gr+tile*bytes_per_256(gt);const uint8_t* uptr=ur+tile*bytes_per_256(ut);
   uint32_t gate_packet=0,up_packet=0;
   uint32_t gate_iq3[2]={0,0},up_iq3[2]={0,0};
   if constexpr(CooperativeIQ3&&GT==21){gate_iq3[0]=cooperative_iq3_word<Mode==4>(gp,lane,0,lut);gate_iq3[1]=cooperative_iq3_word<Mode==4>(gp,lane,1,lut);}
   if constexpr(CooperativeIQ3&&UT==21){up_iq3[0]=cooperative_iq3_word<Mode==4>(uptr,lane,0,lut);up_iq3[1]=cooperative_iq3_word<Mode==4>(uptr,lane,1,lut);}
   if constexpr(Cooperative&&GT==23){if constexpr(Mode==7){gate_packet=pre_gate;if(tile+1<20)pre_gate=iq4_packet<0>(gp+bytes_per_256(GT),lane);}else gate_packet=iq4_packet<Mode>(gp,lane);}
   if constexpr(Cooperative&&UT==23){if constexpr(Mode==7){up_packet=pre_up;if(tile+1<20)pre_up=iq4_packet<0>(uptr+bytes_per_256(UT),lane);}else up_packet=iq4_packet<Mode>(uptr,lane);}
   #pragma unroll
   for(int r=0;r<4;r++){int j=lane+32*r;
    gs[r]=fmaf(CooperativeIQ3&&GT==21?cooperative_iq3_value(gp,gate_iq3[0],lane,r,0):(PackedGrid&&GT==21?iq3s_word_value(gp,j):(gt==23?(Cooperative&&GT==23?cooperative_iq4_value<CachedLevels,Mode>(*(const Block*)gp,gate_packet,cached_level,lane,r,0):weight(*(const Block*)gp,j,Inline)):element_k(gp,j,gt))),x[tile*256+j],gs[r]);
    gs[r]=fmaf(CooperativeIQ3&&GT==21?cooperative_iq3_value(gp,gate_iq3[1],lane,r,1):(PackedGrid&&GT==21?iq3s_word_value(gp,j+128):(gt==23?(Cooperative&&GT==23?cooperative_iq4_value<CachedLevels,Mode>(*(const Block*)gp,gate_packet,cached_level,lane,r,1):weight(*(const Block*)gp,j+128,Inline)):element_k(gp,j+128,gt))),x[tile*256+j+128],gs[r]);
    us[r]=fmaf(CooperativeIQ3&&UT==21?cooperative_iq3_value(uptr,up_iq3[0],lane,r,0):(ut==23?(Cooperative&&UT==23?cooperative_iq4_value<CachedLevels,Mode>(*(const Block*)uptr,up_packet,cached_level,lane,r,0):weight(*(const Block*)uptr,j,Inline)):element_k(uptr,j,ut)),x[tile*256+j],us[r]);
    us[r]=fmaf(CooperativeIQ3&&UT==21?cooperative_iq3_value(uptr,up_iq3[1],lane,r,1):(ut==23?(Cooperative&&UT==23?cooperative_iq4_value<CachedLevels,Mode>(*(const Block*)uptr,up_packet,cached_level,lane,r,1):weight(*(const Block*)uptr,j+128,Inline)):element_k(uptr,j+128,ut)),x[tile*256+j+128],us[r]);
   }
  }
  float g=(gs[0]+gs[2])+(gs[1]+gs[3]),u=(us[0]+us[2])+(us[1]+us[3]);
  for(int n=16;n>0;n/=2){g+=__shfl_down(g,n,32);u+=__shfl_down(u,n,32);}
  if(lane==0)act[row]=g*scalar_sigmoid(g)*u;
 }
}
template<int G,int U> const void* ai_tuned_gate_pointer(int mode){
 switch(mode){
 #define AI_TUNED_GATE_PTR(M) case M:return (const void*)scalar_warp_gate_up<G,U,false,false,true,true,true,true,M>;
 AI_TUNED_GATE_PTR(1) AI_TUNED_GATE_PTR(2) AI_TUNED_GATE_PTR(3) AI_TUNED_GATE_PTR(4) AI_TUNED_GATE_PTR(5) AI_TUNED_GATE_PTR(6) AI_TUNED_GATE_PTR(7)
 #undef AI_TUNED_GATE_PTR
 default:return nullptr;
 }
}
template<int G,int U> void ai_launch_tuned_gate(int mode,const void* gate,const void* up,float* x,float* act,int gt,int ut,int blocks){
 // Avoid thousands of empty blocks when larger row groups cover the same rows.
 int count=std::min(blocks,(17408+ai_tune_rows(mode)-1)/ai_tune_rows(mode));
 switch(mode){
 #define AI_TUNED_GATE_LAUNCH(M) case M:hipLaunchKernelGGL((scalar_warp_gate_up<G,U,false,false,true,true,true,true,M>),dim3(count),dim3(32*ai_tune_rows(M)),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,x,act,gt,ut);break;
 AI_TUNED_GATE_LAUNCH(1) AI_TUNED_GATE_LAUNCH(2) AI_TUNED_GATE_LAUNCH(3) AI_TUNED_GATE_LAUNCH(4) AI_TUNED_GATE_LAUNCH(5) AI_TUNED_GATE_LAUNCH(6) AI_TUNED_GATE_LAUNCH(7)
 #undef AI_TUNED_GATE_LAUNCH
 }
}
template<int G,int U>
void launch_gate_up(const void* gate,const void* up,float* input,float* act,int gt,int ut,int blocks,int warp_reduce,int warp_rows){
 const char* flag=std::getenv("AMD_INFER_INLINE_IQ4_LEVELS");bool inlined=flag&&flag[0]=='1';
  const char* scalar=std::getenv("AMD_INFER_SCALAR_FFN");
  if(warp_rows&&scalar&&scalar[0]=='1'){
   const char* uniform=std::getenv("AMD_INFER_UNIFORM_ROW");
   if(uniform&&uniform[0]=='1'){
    if constexpr((G==21||G==23)&&(U==21||U==23)){if(!inlined&&ai_regroup_requested()){ai_launch_regrouped_gate<G,U>(gate,up,input,act,blocks);return;}}

    int tune=ai_tune_mode();const char* cg=std::getenv("AMD_INFER_COOP_GATE_UP"),*cl=std::getenv("AMD_INFER_COOP_IQ4_LEVELS"),*ci=std::getenv("AMD_INFER_COOP_IQ3S");
    if constexpr(G>=0&&U>=0){if(!inlined&&tune&&cg&&cg[0]=='1'&&cl&&cl[0]=='1'&&ci&&ci[0]=='1'){ai_launch_tuned_gate<G,U>(tune,gate,up,input,act,gt,ut,blocks);return;}}

    const char* iq3=std::getenv("AMD_INFER_COOP_IQ3S");
    if constexpr(G==21||U==21){if(!inlined&&iq3&&iq3[0]=='1'){
     const char* packet=std::getenv("AMD_INFER_COOP_GATE_UP"),*cached=std::getenv("AMD_INFER_COOP_IQ4_LEVELS");
     if(packet&&packet[0]=='1'&&cached&&cached[0]=='1'){
      hipLaunchKernelGGL((scalar_warp_gate_up<G,U,false,false,true,true,true,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);
     }else{hipLaunchKernelGGL((scalar_warp_gate_up<G,U,false,false,true,false,false,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}return;
    }}
    const char* coop=std::getenv("AMD_INFER_COOP_GATE_UP");
    if constexpr(G==23||U==23){
     if(!inlined&&coop&&coop[0]=='1'){
      const char* cache=std::getenv("AMD_INFER_COOP_IQ4_LEVELS");
      if(cache&&cache[0]=='1'){
       hipLaunchKernelGGL((scalar_warp_gate_up<G,U,false,false,true,true,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);return;
      }
      hipLaunchKernelGGL((scalar_warp_gate_up<G,U,false,false,true,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);return;
     }
    }
    if(inlined){hipLaunchKernelGGL((scalar_warp_gate_up<G,U,true,false,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}
    else{hipLaunchKernelGGL((scalar_warp_gate_up<G,U,false,false,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}return;
   }

   const char* packed=std::getenv("AMD_INFER_IQ3S_PACKED_GRID");
   if(G==21&&U==23&&packed&&packed[0]=='1'){
    if(inlined){hipLaunchKernelGGL((scalar_warp_gate_up<G,U,true,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}
    else{hipLaunchKernelGGL((scalar_warp_gate_up<G,U,false,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}return;
   }
  if(inlined){hipLaunchKernelGGL((scalar_warp_gate_up<G,U,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}
  else{hipLaunchKernelGGL((scalar_warp_gate_up<G,U,false>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}return;
 }
 if(inlined){
  if(warp_rows){hipLaunchKernelGGL((warp_row_gate_up<G,U,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}
  else{hipLaunchKernelGGL((persistent_gate_up<G,U,true>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut,warp_reduce);}
 }else{
  if(warp_rows){hipLaunchKernelGGL((warp_row_gate_up<G,U,false>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut);}
  else{hipLaunchKernelGGL((persistent_gate_up<G,U,false>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)gate,(const uint8_t*)up,input,act,gt,ut,warp_reduce);}
 }
}
template<int G,int U,bool UniformRow=false,bool Cooperative=false,bool CachedLevels=false,bool CooperativeIQ3=false> int gate_resources(int* values){
 hipFuncAttributes a{};hipDeviceProp_t p{};int blocks=0,device=0;
 int e=hipGetDevice(&device);if(e)return e;e=hipGetDeviceProperties(&p,device);if(e)return e;
 const void* kernel=(const void*)scalar_warp_gate_up<G,U,false,false,UniformRow,Cooperative,CachedLevels,CooperativeIQ3>;
 e=hipFuncGetAttributes(&a,kernel);if(e)return e;
 e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0);if(e)return e;
 values[0]=a.numRegs;values[1]=int(a.sharedSizeBytes);values[2]=int(a.localSizeBytes);
 values[3]=blocks;values[4]=p.maxThreadsPerMultiProcessor;values[5]=p.multiProcessorCount;
 values[6]=p.warpSize;values[7]=a.maxThreadsPerBlock;return 0;
}
extern "C" int ai_gate_resources(int gt,int ut,int* values){
 if(ai_regroup_requested()){const void* k=ai_regrouped_gate_pointer(gt,ut);if(k){hipFuncAttributes a{};hipDeviceProp_t p{};int d=0,b=0,e=hipGetDevice(&d);if(e)return e;e=hipGetDeviceProperties(&p,d);if(e)return e;e=hipFuncGetAttributes(&a,k);if(e)return e;e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&b,k,128,0);if(e)return e;values[0]=a.numRegs;values[1]=a.sharedSizeBytes;values[2]=a.localSizeBytes;values[3]=b;values[4]=p.maxThreadsPerMultiProcessor;values[5]=p.multiProcessorCount;values[6]=p.warpSize;values[7]=a.maxThreadsPerBlock;return 0;}}

 const char* flag=std::getenv("AMD_INFER_UNIFORM_ROW");bool uniform=flag&&flag[0]=='1';
 const char* coop=std::getenv("AMD_INFER_COOP_GATE_UP");bool cooperative=coop&&coop[0]=='1';
 #define AI_RESOURCE(G,U) if(gt==G&&ut==U){int tune=ai_tune_mode();if(tune&&uniform&&cooperative){const char* cl=std::getenv("AMD_INFER_COOP_IQ4_LEVELS"),*ci=std::getenv("AMD_INFER_COOP_IQ3S");if(cl&&cl[0]=='1'&&ci&&ci[0]=='1'){hipFuncAttributes a{};hipDeviceProp_t p{};int d=0,b=0;const void* k=ai_tuned_gate_pointer<G,U>(tune);int e=hipGetDevice(&d);if(e)return e;e=hipGetDeviceProperties(&p,d);if(e)return e;e=hipFuncGetAttributes(&a,k);if(e)return e;e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&b,k,32*ai_tune_rows(tune),0);if(e)return e;values[0]=a.numRegs;values[1]=a.sharedSizeBytes;values[2]=a.localSizeBytes;values[3]=b;values[4]=p.maxThreadsPerMultiProcessor;values[5]=p.multiProcessorCount;values[6]=p.warpSize;values[7]=a.maxThreadsPerBlock;return 0;}}if constexpr(G==21||U==21){const char* iq3=std::getenv("AMD_INFER_COOP_IQ3S");if(uniform&&iq3&&iq3[0]=='1'){  const char* cached=std::getenv("AMD_INFER_COOP_IQ4_LEVELS");if(cooperative&&cached&&cached[0]=='1')return gate_resources<G,U,true,true,true,true>(values);return gate_resources<G,U,true,false,false,true>(values);}}if constexpr(G==23||U==23){if(uniform&&cooperative){const char* cached=std::getenv("AMD_INFER_COOP_IQ4_LEVELS");if(cached&&cached[0]=='1')return gate_resources<G,U,true,true,true>(values);return gate_resources<G,U,true,true>(values);}}return uniform?gate_resources<G,U,true>(values):gate_resources<G,U,false>(values);}
 AI_RESOURCE(23,23) AI_RESOURCE(21,23) AI_RESOURCE(21,21) AI_RESOURCE(18,21)
 AI_RESOURCE(23,12) AI_RESOURCE(13,13) AI_RESOURCE(18,22) AI_RESOURCE(11,21)
 AI_RESOURCE(22,22) AI_RESOURCE(22,18) AI_RESOURCE(17,22) AI_RESOURCE(17,21)
 AI_RESOURCE(21,11) AI_RESOURCE(18,18) AI_RESOURCE(23,21) AI_RESOURCE(11,11)
 #undef AI_RESOURCE
 return hipErrorInvalidValue;
}
extern "C" int ai_linear_resources(int type,int* values){
 hipFuncAttributes a{};hipDeviceProp_t p{};int blocks=0,device=0;
 int e=hipGetDevice(&device);if(e)return e;e=hipGetDeviceProperties(&p,device);if(e)return e;
 const void* kernel=type==23?(const void*)iq4_linear<false>:(const void*)parallel_k_linear;
 e=hipFuncGetAttributes(&a,kernel);if(e)return e;
 e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0);if(e)return e;
 values[0]=a.numRegs;values[1]=int(a.sharedSizeBytes);values[2]=int(a.localSizeBytes);
 values[3]=blocks;values[4]=p.maxThreadsPerMultiProcessor;values[5]=p.multiProcessorCount;
 values[6]=p.warpSize;values[7]=a.maxThreadsPerBlock;return 0;
}
extern "C" int ai_warp_linear_resources(int type,int* values){
 hipFuncAttributes a{};hipDeviceProp_t p{};int blocks=0,device=0;
 int e=hipGetDevice(&device);if(e)return e;e=hipGetDeviceProperties(&p,device);if(e)return e;
 const void* kernel=nullptr;
 #define AI_RESOURCE_WARP(T) case T:kernel=(const void*)warp_quant_linear<T>;break;
 const char* all=std::getenv("AMD_INFER_COOP_QUANT_GEMV");
 if(ai_regroup_requested()&&type==23)kernel=(const void*)regrouped_probe<23>;
 else if(ai_regroup_requested()&&type==21)kernel=(const void*)regrouped_probe<21>;
 else if(all&&all[0]=='1'&&type==23)kernel=ai_tuned_linear_pointer<23>(ai_tune_mode());
 else if(all&&all[0]=='1'&&type==21)kernel=ai_tuned_linear_pointer<21>(ai_tune_mode());
 else switch(type){AI_RESOURCE_WARP(0) AI_RESOURCE_WARP(8) AI_RESOURCE_WARP(10) AI_RESOURCE_WARP(11) AI_RESOURCE_WARP(12) AI_RESOURCE_WARP(13) AI_RESOURCE_WARP(14) AI_RESOURCE_WARP(17) AI_RESOURCE_WARP(18) AI_RESOURCE_WARP(20) AI_RESOURCE_WARP(21) AI_RESOURCE_WARP(22) AI_RESOURCE_WARP(23) default:return hipErrorInvalidValue;}
 #undef AI_RESOURCE_WARP
 e=hipFuncGetAttributes(&a,kernel);if(e)return e;
 e=hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,(type==21||type==23)&&all&&all[0]=='1'?32*ai_tune_rows(ai_tune_mode()):128,0);if(e)return e;
 values[0]=a.numRegs;values[1]=int(a.sharedSizeBytes);values[2]=int(a.localSizeBytes);
 values[3]=blocks;values[4]=p.maxThreadsPerMultiProcessor;values[5]=p.multiProcessorCount;values[6]=p.warpSize;values[7]=a.maxThreadsPerBlock;return 0;
}
 template<int Type>
 __global__ void specialized_down(const uint8_t* w,const float* x,float* y){
  int row=blockIdx.x,lane=threadIdx.x;const int stride=bytes_per_256(Type);
  const uint8_t* wr=w+size_t(row)*68*stride;float sum=0;
  for(int tile=0;tile<68;tile++){
   const uint8_t* p=wr+tile*stride;
   sum=fmaf(Type==23?weight(*(const Block*)p,lane):element_k(p,lane,Type),x[tile*256+lane],sum);
   sum=fmaf(Type==23?weight(*(const Block*)p,lane+128):element_k(p,lane+128,Type),x[tile*256+lane+128],sum);
  }

  __shared__ float partial[128];partial[lane]=sum;__syncthreads();
  for(int n=64;n>0;n/=2){if(lane<n)partial[lane]+=partial[lane+n];__syncthreads();}
  if(lane==0)y[row]=partial[0];
 }
 int launch_ffn_down(const void* w,const float* x,float* y,int type){
  const char* warp=std::getenv("AMD_INFER_WARP_GEMV");
  if(warp&&warp[0]=='1')return ai_quant_linear(w,x,y,17408,5120,1,type);
  const char* flag=std::getenv("AMD_INFER_DOWN_VARIANT");
  if(!flag||flag[0]=='0')return ai_quant_linear(w,x,y,17408,5120,1,type);
  if(flag[0]!='1')return hipErrorInvalidValue;
  switch(type){
   #define AI_DOWN(T) case T:hipLaunchKernelGGL((specialized_down<T>),dim3(5120),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y);break;
   AI_DOWN(0) AI_DOWN(8) AI_DOWN(10) AI_DOWN(11) AI_DOWN(12) AI_DOWN(13) AI_DOWN(14) AI_DOWN(17) AI_DOWN(18) AI_DOWN(20) AI_DOWN(21) AI_DOWN(22) AI_DOWN(23)
   #undef AI_DOWN
   default:return hipErrorInvalidValue;
  }
  return hipGetLastError();
 }
 extern "C" {
  struct AITiming {hipEvent_t begin,end,norm,gate;};
 int ai_ffn_raw(const void* gate,const void* up,const void* down,int gate_type,int up_type,int down_type,const float* norm,float* input,float* output,int persistent_blocks,int specialize,int warp_reduce,int warp_rows,void* timing){
  int e=ai_rms(input,norm,input,5120);if(e)return e;
  if(timing){e=hipEventRecord(((AITiming*)timing)->norm,ai_stream());if(e)return e;}
 if(persistent_blocks){
  if(persistent_blocks<128||persistent_blocks>4096)return hipErrorInvalidValue;
  // The actual sixteen gate/up pairs cover all 64 main-model layers.
  #define AI_PAIR(G,U) if(specialize&&gate_type==G&&up_type==U){launch_gate_up<G,U>(gate,up,input,output+17408,gate_type,up_type,persistent_blocks,warp_reduce,warp_rows);}else
  AI_PAIR(23,23) AI_PAIR(21,23) AI_PAIR(21,21) AI_PAIR(18,21)
  AI_PAIR(23,12) AI_PAIR(13,13) AI_PAIR(18,22) AI_PAIR(11,21)
  AI_PAIR(22,22) AI_PAIR(22,18) AI_PAIR(17,22) AI_PAIR(17,21)
  AI_PAIR(21,11) AI_PAIR(18,18) AI_PAIR(23,21) AI_PAIR(11,11)
  {launch_gate_up<-1,-1>(gate,up,input,output+17408,gate_type,up_type,persistent_blocks,warp_reduce,warp_rows);}
  #undef AI_PAIR
    e=hipGetLastError();if(e)return e;
    if(timing){e=hipEventRecord(((AITiming*)timing)->gate,ai_stream());if(e)return e;}
   return launch_ffn_down(down,output+17408,output,down_type);
 }
 e=ai_quant_linear(gate,input,output,5120,17408,1,gate_type);if(e)return e;
 e=ai_quant_linear(up,input,output+17408,5120,17408,1,up_type);if(e)return e;
   hipLaunchKernelGGL(swiglu_kernel,dim3((17408+255)/256),dim3(256),0,ai_stream(),output,output+17408,input,17408);e=hipGetLastError();if(e)return e;
   if(timing){e=hipEventRecord(((AITiming*)timing)->gate,ai_stream());if(e)return e;}
  return launch_ffn_down(down,input,output,down_type);
}
int ai_ffn(const void* gate,const void* up,const void* down,int gate_type,int up_type,int down_type,const float* norm,float* input,float* output,int persistent_blocks,int specialize,int warp_reduce,int warp_rows,void* timing){
 auto body=[&](){return ai_ffn_raw(gate,up,down,gate_type,up_type,down_type,norm,input,output,persistent_blocks,specialize,warp_reduce,warp_rows,timing);};
 if(!ai_graph_requested()||timing)return body();
 std::vector<uintptr_t> key{2,ai_math_key(),ai_pointer(gate),ai_pointer(up),ai_pointer(down),uintptr_t(gate_type),uintptr_t(up_type),uintptr_t(down_type),ai_pointer(norm),ai_pointer(input),ai_pointer(output),uintptr_t(persistent_blocks),uintptr_t(specialize),uintptr_t(warp_reduce),uintptr_t(warp_rows)};
 return ai_graph_dispatch(key,body);
}
__global__ void read_bandwidth_kernel(const float* input,float* output,int count){
 int lane=threadIdx.x;float sum=0;for(int i=blockIdx.x*blockDim.x+lane;i<count;i+=gridDim.x*blockDim.x)sum+=input[i];
 __shared__ float sums[256];sums[lane]=sum;__syncthreads();for(int n=128;n>0;n/=2){if(lane<n)sums[lane]+=sums[lane+n];__syncthreads();}if(lane==0)output[blockIdx.x]=sums[0];
}
int ai_read_bandwidth(const float* input,float* output,int count){hipLaunchKernelGGL(read_bandwidth_kernel,dim3(2048),dim3(256),0,ai_stream(),input,output,count);return hipGetLastError();}
 int ai_timing_new(void** out){auto* p=new AITiming{};hipEvent_t* events[]={&p->begin,&p->end,&p->norm,&p->gate};for(int i=0;i<4;i++){auto e=hipEventCreate(events[i]);if(e){for(int j=0;j<i;j++)hipEventDestroy(*events[j]);delete p;return e;}}*out=p;return hipSuccess;}
 int ai_timing_free(void* ptr){auto* p=(AITiming*)ptr;int error=0;for(auto event:{p->begin,p->end,p->norm,p->gate}){int e=hipEventDestroy(event);if(!error)error=e;}delete p;return error;}
 int ai_timing_ffn_parts(void* ptr,float* norm,float* gate,float* down){auto* p=(AITiming*)ptr;int e=hipEventElapsedTime(norm,p->begin,p->norm);if(e)return e;e=hipEventElapsedTime(gate,p->norm,p->gate);if(e)return e;return hipEventElapsedTime(down,p->gate,p->end);}
int ai_timing_begin(void* ptr){return hipEventRecord(((AITiming*)ptr)->begin,ai_stream());}
int ai_timing_end(void* ptr,float* ms){auto* p=(AITiming*)ptr;auto e=hipEventRecord(p->end,ai_stream());if(e!=hipSuccess)return e;e=hipEventSynchronize(p->end);if(e!=hipSuccess)return e;return hipEventElapsedTime(ms,p->begin,p->end);}
int ai_timed_linear(void* ptr,const void* w,const float* x,float* y,int cols,int rows,int type,float* ms){
 auto* p=(AITiming*)ptr;auto e=hipEventRecord(p->begin,ai_stream());if(e!=hipSuccess)return e;
 int launch=ai_quant_linear(w,x,y,cols,rows,1,type);if(launch)return launch;
 e=hipEventRecord(p->end,ai_stream());if(e!=hipSuccess)return e;e=hipEventSynchronize(p->end);if(e!=hipSuccess)return e;return hipEventElapsedTime(ms,p->begin,p->end);
}
}

// Phase11 decision probe:FP32 grouped dot;not selected by ai_quant_linear.
// Each lane owns eight contiguous values of a32-value scale group.
template<int Kind> __global__ void regrouped_probe(const uint8_t* w,const float* x,float* y,int cols,int rows){
 int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x*4+warp;if(row>=rows)return;
 int group=lane/4,segment=lane%4;const uint8_t* wr=w+size_t(row)*(cols/256)*bytes_per_256(Kind);
 float cached=levels[lane&15],sum=0;
 for(int tile=0;tile<cols/256;tile++){
  const uint8_t* p=wr+tile*bytes_per_256(Kind);int j=group*32+segment*8;
  float4 a=((const float4*)(x+tile*256+j))[0],b=((const float4*)(x+tile*256+j))[1];
  uint32_t words[2];int signs=0;float scale=0;
  if constexpr(Kind==23){auto* q=(const Block*)p;uint2 v=((const uint2*)(p+8+group*16))[segment&1];words[0]=v.x;words[1]=v.y;
   int sc=((q->scales_l[group/2]>>(4*(group%2)))&15)|(((q->scales_h>>(2*group))&3)<<4);scale=ai_half(q->d)*(sc-32);
  }else{auto* q=(const block_iq3_s*)p;signs=q->signs[group*4+segment];
   words[0]=iq3s_grid[q->qs[group*8+2*segment]|(((q->qh[group]>>(2*segment))&1)<<8)];
   words[1]=iq3s_grid[q->qs[group*8+2*segment+1]|(((q->qh[group]>>(2*segment+1))&1)<<8)];
   scale=ai_half(q->d)*(1+2*((q->scales[group/2]>>(4*(group%2)))&15));
  }
  float inner=0;
  #pragma unroll
  for(int i=0;i<8;i++){
   float xv=i==0?a.x:i==1?a.y:i==2?a.z:i==3?a.w:i==4?b.x:i==5?b.y:i==6?b.z:b.w;
   float value;
   if constexpr(Kind==23){int code=(words[i/4]>>(8*(i%4)+4*(segment/2)))&15;value=__shfl(cached,code,32);}
   else value=float((words[i/4]>>(8*(i%4)))&255u)*(signs&(1<<i)?-1.f:1.f);
   inner=fmaf(value,xv,inner);
  }sum=fmaf(scale,inner,sum);
 }
 for(int n=16;n>0;n/=2)sum+=__shfl_down(sum,n,32);if(lane==0)y[row]=sum;
}
extern "C" int ai_timed_regrouped_probe(void* ptr,const void* w,const float* x,float* y,int cols,int rows,int type,float* ms){
 if(cols<=0||cols%256||rows<=0||(type!=21&&type!=23))return hipErrorInvalidValue;
 auto* p=(AITiming*)ptr;int e=hipEventRecord(p->begin,ai_stream());if(e)return e;
 if(type==23){hipLaunchKernelGGL((regrouped_probe<23>),dim3((rows+3)/4),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows);}
 else{hipLaunchKernelGGL((regrouped_probe<21>),dim3((rows+3)/4),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows);}
 e=hipGetLastError();if(e)return e;e=hipEventRecord(p->end,ai_stream());if(e)return e;e=hipEventSynchronize(p->end);if(e)return e;return hipEventElapsedTime(ms,p->begin,p->end);
}

struct AIGroupedDot {float inner,scale;};
template<int Kind> __device__ AIGroupedDot ai_grouped_dot(const uint8_t* p,float4 a,float4 b,int lane,float cached){
 int group=lane/4,segment=lane%4;uint32_t words[2];int signs=0;float scale;
 if constexpr(Kind==23){auto* q=(const Block*)p;uint2 v=((const uint2*)(p+8+group*16))[segment&1];words[0]=v.x;words[1]=v.y;
  int sc=((q->scales_l[group/2]>>(4*(group%2)))&15)|(((q->scales_h>>(2*group))&3)<<4);scale=ai_half(q->d)*(sc-32);
 }else{auto* q=(const block_iq3_s*)p;signs=q->signs[group*4+segment];
  words[0]=iq3s_grid[q->qs[group*8+2*segment]|(((q->qh[group]>>(2*segment))&1)<<8)];
  words[1]=iq3s_grid[q->qs[group*8+2*segment+1]|(((q->qh[group]>>(2*segment+1))&1)<<8)];
  scale=ai_half(q->d)*(1+2*((q->scales[group/2]>>(4*(group%2)))&15));
 }float inner=0;
 #pragma unroll
 for(int i=0;i<8;i++){
  float xv=i==0?a.x:i==1?a.y:i==2?a.z:i==3?a.w:i==4?b.x:i==5?b.y:i==6?b.z:b.w;float value;
  if constexpr(Kind==23){int code=(words[i/4]>>(8*(i%4)+4*(segment/2)))&15;value=__shfl(cached,code,32);}
  else value=float((words[i/4]>>(8*(i%4)))&255u)*(signs&(1<<i)?-1.f:1.f);
  inner=fmaf(value,xv,inner);
 }return {inner,scale};
}
template<int G,int U> __global__ void regrouped_gate(const uint8_t* gate,const uint8_t* up,const float* x,float* act){
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 #if defined(__HIP_DEVICE_COMPILE__)
 warp=__builtin_amdgcn_readfirstlane(warp);
 #endif
 float cached=levels[lane&15];int j=(lane/4)*32+(lane%4)*8;
 for(int row=blockIdx.x*4+warp;row<17408;row+=gridDim.x*4){
  const uint8_t* gr=gate+size_t(row)*20*bytes_per_256(G),*ur=up+size_t(row)*20*bytes_per_256(U);float g=0,u=0;
  for(int tile=0;tile<20;tile++){
   float4 a=((const float4*)(x+tile*256+j))[0],b=((const float4*)(x+tile*256+j))[1];
   auto dg=ai_grouped_dot<G>(gr+tile*bytes_per_256(G),a,b,lane,cached);
   auto du=ai_grouped_dot<U>(ur+tile*bytes_per_256(U),a,b,lane,cached);
   g=fmaf(dg.scale,dg.inner,g);u=fmaf(du.scale,du.inner,u);
  }
  for(int n=16;n>0;n/=2){g+=__shfl_down(g,n,32);u+=__shfl_down(u,n,32);}if(lane==0)act[row]=g*scalar_sigmoid(g)*u;
 }
}
template<int G,int U> void ai_launch_regrouped_gate(const void* g,const void* u,float* x,float* act,int blocks){
 hipLaunchKernelGGL((regrouped_gate<G,U>),dim3(blocks),dim3(128),0,ai_stream(),(const uint8_t*)g,(const uint8_t*)u,x,act);
}
const void* ai_regrouped_gate_pointer(int g,int u){
 if(g==23&&u==23)return (const void*)regrouped_gate<23,23>;
 if(g==21&&u==23)return (const void*)regrouped_gate<21,23>;
 if(g==21&&u==21)return (const void*)regrouped_gate<21,21>;
 if(g==23&&u==21)return (const void*)regrouped_gate<23,21>;return nullptr;
}
int ai_launch_regrouped_model(const void* w,const float* x,float* y,int cols,int rows,int kind){
 if(kind==23){hipLaunchKernelGGL((regrouped_probe<23>),dim3((rows+3)/4),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows);}
 else{hipLaunchKernelGGL((regrouped_probe<21>),dim3((rows+3)/4),dim3(128),0,ai_stream(),(const uint8_t*)w,x,y,cols,rows);}return hipGetLastError();
}
