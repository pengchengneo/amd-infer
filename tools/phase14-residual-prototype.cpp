#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" int ai_alloc(void**,size_t),ai_free(void*),ai_h2d(void*,const void*,size_t),ai_d2h(void*,const void*,size_t),ai_d2d(void*,const void*,size_t),ai_residual(float*,const float*),ai_rms(const float*,const float*,float*,int),ai_sync(),ai_gate_resources(int,int,int*),ai_warp_linear_resources(int,int*);
#define OK(call) do { int error=(call);if(error){std::fprintf(stderr,"failure %s=%d\n",#call,error);return 2;} } while(0)
// One block owns every updated hidden element and the full norm reduction.
// The reduction indexing, FP64 tree, epsilon and final FP32 products match
// rms_stable_kernel. No block waits for another block or a future launch.
__global__ void residual_rms_prototype(float* hidden,const float* add,const float* weights,float* normalized){
    __shared__ double partial[256];int lane=threadIdx.x;double s=0;
    for(int i=lane;i<5120;i+=256){float updated=__fadd_rn(hidden[i],add[i]);hidden[i]=updated;double v=updated;s+=v*v;}
    partial[lane]=s;__syncthreads();
    for(int d=128;d;d/=2){if(lane<d)partial[lane]+=partial[lane+d];__syncthreads();}
    float inv=float(1.0/sqrt(partial[0]/double(5120)+double(1e-6f)));
    for(int i=lane;i<5120;i+=256)normalized[i]=hidden[i]*inv*weights[i];
}
int main(){
    int device=0;OK(hipGetDevice(&device));hipDeviceProp_t p{};OK(hipGetDeviceProperties(&p,device));
    int cooperative=0,multi=0;OK(hipDeviceGetAttribute(&cooperative,hipDeviceAttributeCooperativeLaunch,device));OK(hipDeviceGetAttribute(&multi,hipDeviceAttributeCooperativeMultiDeviceLaunch,device));
    hipFuncAttributes attrs{};int active=0;OK(hipFuncGetAttributes(&attrs,(const void*)residual_rms_prototype));OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,residual_rms_prototype,256,0));
    std::printf("CAPS name=%s architecture=%s multiprocessors=%d warp=%d max_threads_block=%d max_threads_mp=%d regs_block=%d shared_block=%zu shared_mp=%zu cooperative=%d cooperative_multi=%d\n",p.name,p.gcnArchName,p.multiProcessorCount,p.warpSize,p.maxThreadsPerBlock,p.maxThreadsPerMultiProcessor,p.regsPerBlock,p.sharedMemPerBlock,p.maxSharedMemoryPerMultiProcessor,cooperative,multi);
    std::printf("RESOURCE residual regs=%d shared=%zu local=%zu active_blocks_mp=%d grid_resident_upper=%d\n",attrs.numRegs,attrs.sharedSizeBytes,attrs.localSizeBytes,active,active*p.multiProcessorCount);
    int pairs[][2]={{23,23},{21,23},{21,21},{18,21},{23,12},{13,13},{18,22},{11,21},{22,22},{22,18},{17,22},{17,21},{21,11},{18,18},{23,21},{11,11}};
    for(auto& pair:pairs){int v[8]{};OK(ai_gate_resources(pair[0],pair[1],v));std::printf("RESOURCE gate_%d_%d regs=%d shared=%d local=%d active_blocks_mp=%d grid_resident_upper=%d\n",pair[0],pair[1],v[0],v[1],v[2],v[3],v[3]*v[5]);}
    for(int type:{23,21}){int v[8]{};OK(ai_warp_linear_resources(type,v));std::printf("RESOURCE linear_%d regs=%d shared=%d local=%d active_blocks_mp=%d grid_resident_upper=%d\n",type,v[0],v[1],v[2],v[3],v[3]*v[5]);}
    constexpr int cols=5120,steps=512;size_t bytes=cols*4;float *hidden,*add,*weights,*out;
    OK(ai_alloc((void**)&hidden,bytes));OK(ai_alloc((void**)&add,bytes));OK(ai_alloc((void**)&weights,bytes));OK(ai_alloc((void**)&out,bytes));hipEvent_t begin,end;OK(hipEventCreate(&begin));OK(hipEventCreate(&end));
    for(int pattern=0;pattern<4;pattern++){
        std::vector<float> initial(cols),addition(cols),norm(cols);
        for(int i=0;i<cols;i++){
            if(pattern==0){initial[i]=.1f*std::sin(i*.01f);addition[i]=.001f*std::cos(i*.03f);}
            if(pattern==1){initial[i]=(i%2?1.f:-1.f)*std::ldexp(1.f,(i%31)-15);addition[i]=std::ldexp((i%3?1.f:-1.f),(i%25)-20);}
            if(pattern==2){initial[i]=0;addition[i]=0;}
            if(pattern==3){initial[i]=std::ldexp(float((i%7)-3),-120);addition[i]=std::ldexp(float((i%5)-2),-121);}
            norm[i]=.9f+.01f*std::sin(i*.11f);
        }
        OK(ai_h2d(add,addition.data(),bytes));OK(ai_h2d(weights,norm.data(),bytes));std::vector<float> expected_h,expected_out;
        for(int trial=0;trial<12;trial++){
            bool candidate=trial%6==1||trial%6==2||trial%6==5;OK(ai_h2d(hidden,initial.data(),bytes));OK(ai_sync());OK(hipEventRecord(begin,0));
            for(int j=0;j<steps;j++){
                if(candidate){hipLaunchKernelGGL(residual_rms_prototype,dim3(1),dim3(256),0,0,hidden,add,weights,out);OK(hipGetLastError());}
                else{OK(ai_residual(hidden,add));OK(ai_d2d(out,hidden,bytes));OK(ai_rms(out,weights,out,cols));}
            }
            OK(hipEventRecord(end,0));OK(hipEventSynchronize(end));float ms=0;OK(hipEventElapsedTime(&ms,begin,end));std::vector<float> h(cols),n(cols);OK(ai_d2h(h.data(),hidden,bytes));OK(ai_d2h(n.data(),out,bytes));
            if(trial==0){expected_h=h;expected_out=n;}
            else if(std::memcmp(expected_h.data(),h.data(),bytes)||std::memcmp(expected_out.data(),n.data(),bytes)){std::fprintf(stderr,"bit difference pattern=%d trial=%d\n",pattern,trial);return 3;}
            for(float value:n)if(!std::isfinite(value)){std::fprintf(stderr,"nonfinite output\n");return 4;}
            std::printf("RESIDUAL_PROFILE pattern=%d trial=%d candidate=%d chain_us=%.9g steps=%d bits_same=1\n",pattern,trial,candidate,ms*1000/steps,steps);std::fflush(stdout);
        }
    }
    OK(hipEventDestroy(begin));OK(hipEventDestroy(end));OK(ai_free(hidden));OK(ai_free(add));OK(ai_free(weights));OK(ai_free(out));return 0;
}
