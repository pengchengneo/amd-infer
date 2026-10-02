#pragma once
// Diagnostic event windows; not pure device execution or uninstrumented E2E.
struct AIGdnProfile {
    bool enabled{};int status{};hipEvent_t events[10]{};uint64_t ns[9]{},calls[9]{};
    AIGdnProfile(){const char* flag=std::getenv("AMD_INFER_PROFILE_GDN_PARTS");enabled=flag&&flag[0]=='1';
        if(enabled)for(auto& event:events){status=hipEventCreate(&event);if(status)break;}}
    ~AIGdnProfile(){for(auto event:events)if(event)hipEventDestroy(event);}
};
inline AIGdnProfile& ai_gdn_profile(){static thread_local AIGdnProfile p;return p;}
inline int ai_gdn_mark(int slot){auto& p=ai_gdn_profile();if(p.status)return p.status;return p.enabled?hipEventRecord(p.events[slot],ai_stream()):0;}
inline int ai_gdn_finish(){auto& p=ai_gdn_profile();if(!p.enabled)return 0;
    int e=hipEventSynchronize(p.events[9]);if(e)return e;
    for(int i=0;i<9;i++){float ms=0;e=hipEventElapsedTime(&ms,p.events[i],p.events[i+1]);if(e)return e;p.ns[i]+=uint64_t(double(ms)*1e6);p.calls[i]++;}return 0;}
extern "C" int ai_gdn_profile_stats(uint64_t* values,int reset){auto& p=ai_gdn_profile();
    for(int i=0;i<9;i++){values[2*i]=p.ns[i];values[2*i+1]=p.calls[i];if(reset)p.ns[i]=p.calls[i]=0;}return p.status;}
