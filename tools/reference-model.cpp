// Pinned llama.cpp reference only. This is not AMDInfer's execution engine.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
struct Trace {std::string prefix;};
static bool trace_tensor(ggml_tensor* t,bool ask,void* data){
    std::string name=ggml_get_name(t);bool selected=name=="l_out-0"||name=="result_norm"||name=="linear_attn_qkv_mixed-0";
    if(ask)return selected;if(!selected||t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t))return true;
    auto* trace=(Trace*)data;size_t size=ggml_nbytes(t);std::vector<char> bytes(size);ggml_backend_tensor_get(t,bytes.data(),0,size);std::ofstream out(trace->prefix+"."+name+".f32",std::ios::binary);out.write(bytes.data(),bytes.size());return true;
}
int main(int argc,char** argv){
    if(argc!=5){std::cerr<<"usage: reference-model tokenize|logits MODEL TEXT_FILE OUTPUT_PREFIX\n";return 2;}
    std::string mode=argv[1];if(mode!="tokenize"&&mode!="logits"&&mode!="greedy")return 2;
    std::ifstream input(argv[3],std::ios::binary);if(!input)return 2;
    std::string text((std::istreambuf_iterator<char>(input)),{});
    ggml_backend_load_all();llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.vocab_only=mode=="tokenize";mp.load_mtp=false;mp.use_extra_bufts=false;
    llama_model* model=llama_model_load_from_file(argv[2],mp);if(!model)return 3;
    const llama_vocab* vocab=llama_model_get_vocab(model);
    int n=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
    if(n<0)n=-n;std::vector<llama_token> ids(n);n=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),false,true);if(n<=0)return 4;ids.resize(n);
    std::string prefix=argv[4];std::ofstream tokens(prefix+".tokens.json");tokens<<"[";for(int i=0;i<n;i++){if(i)tokens<<",";tokens<<ids[i];}tokens<<"]\n";tokens.close();
    if(mode=="logits"||mode=="greedy"){
        if(n>128){std::cerr<<"CPU correctness reference capped at 128 tokens\n";return 4;}
        auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=128;cp.n_ubatch=128;cp.n_threads=2;cp.n_threads_batch=2;
        cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        Trace trace{prefix};cp.cb_eval=trace_tensor;cp.cb_eval_user_data=&trace;
        llama_context* ctx=llama_init_from_model(model,cp);if(!ctx)return 5;
        auto batch=llama_batch_get_one(ids.data(),n);if(llama_decode(ctx,batch)!=0)return 6;
        const float* logits=llama_get_logits_ith(ctx,-1);int nv=llama_vocab_n_tokens(vocab);if(!logits)return 7;
        if(mode=="greedy"){
            std::ofstream generated(prefix+".generated.json");generated<<"[";
            for(int step=0;step<8;step++){
                llama_token token=std::max_element(logits,logits+nv)-logits;
                if(step)generated<<",";generated<<token;
                if(step==7)break;
                auto next=llama_batch_get_one(&token,1);if(llama_decode(ctx,next)!=0)return 6;
                logits=llama_get_logits_ith(ctx,-1);if(!logits)return 7;
            }
            generated<<"]\n";
        }
        std::ofstream out(prefix+".logits.f32",std::ios::binary);out.write((const char*)logits,nv*sizeof(float));out.close();
        int top=std::max_element(logits,logits+nv)-logits;std::cout<<"reference tokens="<<n<<" vocab="<<nv<<" top_token="<<top<<"\n";
        llama_free(ctx);
    }else std::cout<<"reference tokens="<<n<<"\n";
    llama_model_free(model);llama_backend_free();return 0;
}
