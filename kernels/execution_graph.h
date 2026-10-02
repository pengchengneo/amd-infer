#pragma once
#include <map>
#include <vector>
#include <chrono>
#include <cstdint>

// Single-thread B1 runtime ownership. All kernel/copy/event dispatch uses this stream.
struct AIGraphEntry { hipGraph_t graph{}; hipGraphExec_t exec{}; };
struct AIExecution {
    hipStream_t stream{};
    int status{};
    std::map<std::vector<uintptr_t>, AIGraphEntry> graphs;
    std::map<uintptr_t,size_t> allocations;
    uint64_t stats[7]{}; // captures, launches, setup ns, failures, skips, invalidations, captured nodes
    AIExecution() {
        const char* flag=std::getenv("AMD_INFER_EXPLICIT_STREAM");
        if(flag&&flag[0]=='1')status=hipStreamCreateWithFlags(&stream,hipStreamNonBlocking);
    }
    void clear() {
        if(!graphs.empty())stats[5]++;
        for(auto& item:graphs) { hipGraphExecDestroy(item.second.exec);hipGraphDestroy(item.second.graph); }
        graphs.clear();
    }
    void invalidate(void* pointer) {
        uintptr_t begin=ai_pointer_value(pointer);
        auto allocation=allocations.find(begin);
        if(allocation==allocations.end()){clear();return;}
        uintptr_t end=begin+allocation->second;
        bool changed=false;
        for(auto it=graphs.begin();it!=graphs.end();) {
            bool used=false;for(auto value:it->first)used|=value>=begin&&value<end;
            if(used){hipGraphExecDestroy(it->second.exec);hipGraphDestroy(it->second.graph);it=graphs.erase(it);changed=true;}
            else ++it;
        }
        if(changed)stats[5]++;
        allocations.erase(allocation);
    }
    static uintptr_t ai_pointer_value(void* pointer){return reinterpret_cast<uintptr_t>(pointer);}
    ~AIExecution() { if(stream)hipStreamSynchronize(stream);clear();if(stream)hipStreamDestroy(stream); }
};
inline AIExecution& ai_execution() { static thread_local AIExecution owned;return owned; }
inline hipStream_t ai_stream() { return ai_execution().stream; }
inline uintptr_t ai_pointer(const void* ptr) { return reinterpret_cast<uintptr_t>(ptr); }
inline uintptr_t ai_math_key() {
    uintptr_t flags=0;int bit=0;
    for(auto name:{"AMD_INFER_STABLE_RMS","AMD_INFER_SCALAR_GDN","AMD_INFER_SCALAR_FFN",
        "AMD_INFER_INLINE_IQ4_LEVELS","AMD_INFER_IQ3S_PACKED_GRID","AMD_INFER_UNIFORM_ROW","AMD_INFER_DOWN_VARIANT","AMD_INFER_WARP_GEMV","AMD_INFER_GDN_HEAD_FUSE","AMD_INFER_COOP_GATE_UP","AMD_INFER_COOP_IQ4_LEVELS","AMD_INFER_COOP_IQ3S","AMD_INFER_COOP_QUANT_GEMV","AMD_INFER_GEMV_TUNE","AMD_INFER_FP32_REGROUP"}) {
        const char* value=std::getenv(name);if(value&&value[0]=='1')flags|=uintptr_t(1)<<bit;bit++;
    }
    const char* grid=std::getenv("AMD_INFER_GDN_HEAD_BLOCKS");
    return flags | (uintptr_t(grid?std::atoi(grid):48)<<16);
}
inline bool ai_graph_requested() {
    const char* flag=std::getenv("AMD_INFER_LOCAL_GRAPHS");
    return flag&&flag[0]=='1'&&ai_stream();
}
template<class Function>
int ai_graph_dispatch(const std::vector<uintptr_t>& key,Function&& body) {
    auto& owned=ai_execution();
    auto found=owned.graphs.find(key);
    if(found!=owned.graphs.end()) { owned.stats[1]++;return hipGraphLaunch(found->second.exec,owned.stream); }
    size_t available=0,total=0;int e=hipMemGetInfo(&available,&total);if(e)return e;
    const size_t reserve=(size_t(2)<<30)+(16<<20);
    if(available<reserve+(16<<20)||owned.graphs.size()>=128) { owned.stats[4]++;return body(); }
    // Capture records work rather than executing it; state changes occur once in replay.
    e=hipStreamSynchronize(owned.stream);if(e)return e;
    auto started=std::chrono::steady_clock::now();
    e=hipStreamBeginCapture(owned.stream,hipStreamCaptureModeThreadLocal);if(e)return e;
    int launch_error=body();AIGraphEntry entry;
    int end_error=hipStreamEndCapture(owned.stream,&entry.graph);
    if(launch_error||end_error) {
        if(entry.graph)hipGraphDestroy(entry.graph);owned.stats[3]++;
        return launch_error?launch_error:end_error;
    }
    size_t nodes=0;e=hipGraphGetNodes(entry.graph,nullptr,&nodes);
    if(!e)e=hipGraphInstantiate(&entry.exec,entry.graph,nullptr,nullptr,0);
    if(e) { hipGraphDestroy(entry.graph);owned.stats[3]++;return e; }
    e=hipMemGetInfo(&available,&total);
    if(e||available<reserve) {
        hipGraphExecDestroy(entry.exec);hipGraphDestroy(entry.graph);
        if(e)return e;owned.stats[4]++;return body();
    }
    owned.stats[0]++;owned.stats[6]+=nodes;
    owned.stats[2]+=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-started).count();
    owned.graphs.emplace(key,entry);owned.stats[1]++;
    return hipGraphLaunch(entry.exec,owned.stream);
}
extern "C" int ai_graph_stats(uint64_t* values,int reset) {
    auto& owned=ai_execution();for(int i=0;i<7;i++)values[i]=owned.stats[i];values[7]=owned.graphs.size();
    if(reset)for(auto& value:owned.stats)value=0;
    return owned.status;
}
