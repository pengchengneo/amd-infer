#pragma once
// Decode budget observer only. Event windows include scheduling gaps.
struct AIAttentionProfile {
 bool enabled{};int status{};hipEvent_t events[8]{};uint64_t ns[7]{},calls[7]{};
 AIAttentionProfile(){const char* flag=std::getenv("AMD_INFER_PROFILE_ATTN_PARTS");enabled=flag&&flag[0]=='1';
  if(enabled)for(auto& e:events){status=hipEventCreate(&e);if(status)break;}}
 ~AIAttentionProfile(){for(auto e:events)if(e)hipEventDestroy(e);}
};
inline AIAttentionProfile& ai_attention_profile(){static thread_local AIAttentionProfile p;return p;}
extern "C" int ai_attention_profile_mark(int slot){auto& p=ai_attention_profile();if(p.status)return p.status;
 if(slot<0||slot>7)return hipErrorInvalidValue;return p.enabled?hipEventRecord(p.events[slot],ai_stream()):0;}
extern "C" int ai_attention_profile_finish(){auto& p=ai_attention_profile();if(!p.enabled)return 0;
 int e=hipEventSynchronize(p.events[7]);if(e)return e;
 for(int i=0;i<7;i++){float ms=0;e=hipEventElapsedTime(&ms,p.events[i],p.events[i+1]);if(e)return e;p.ns[i]+=uint64_t(double(ms)*1e6);p.calls[i]++;}return 0;}
extern "C" int ai_attention_profile_stats(uint64_t* values,int reset){auto& p=ai_attention_profile();
 for(int i=0;i<7;i++){values[2*i]=p.ns[i];values[2*i+1]=p.calls[i];if(reset)p.ns[i]=p.calls[i]=0;}return p.status;}
