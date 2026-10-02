// Project-authored benchmark and Rust-policy sampler port over pinned public ABI.
using System;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Collections.Generic;
public static class Phase13Native {
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] static extern IntPtr llama_get_memory(IntPtr context);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] static extern void llama_memory_clear(IntPtr memory,[MarshalAs(UnmanagedType.I1)]bool data);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] static extern void llama_synchronize(IntPtr context);
 static string Num(double v){return v.ToString("R",CultureInfo.InvariantCulture);}
 static int Compare(float a,int ai,float b,int bi){if(a==0&&b==0){int ab=BitConverter.SingleToInt32Bits(a),bb=BitConverter.SingleToInt32Bits(b);if(ab!=bb)return ab<0?-1:1;}int c=a.CompareTo(b);return c!=0?c:ai.CompareTo(bi);}
 public static int Select(float[] values,bool sampled,ref ulong state) {
  int[] top=new int[40];int used=0,best=0;
  for(int i=0;i<values.Length;i++){
   float x=values[i];if(Single.IsNaN(x)||Single.IsInfinity(x))throw new Exception("Nonfinite logits");
   if(Compare(x,i,values[best],best)>0)best=i;
   if(!sampled)continue;
   if(used==40&&Compare(x,i,values[top[39]],top[39])<=0)continue;
   int pos=Math.Min(used,39);while(pos>0&&Compare(x,i,values[top[pos-1]],top[pos-1])>0){if(pos<40)top[pos]=top[pos-1];pos--;}
   top[pos]=i;if(used<40)used++;
  }
  if(!sampled)return best;
  double[] weights=new double[used];double sum=0;for(int j=0;j<used;j++){weights[j]=Math.Exp(((double)values[top[j]]-(double)values[top[0]])/0.7);sum+=weights[j];}
  ulong z;unchecked{state+=0x9e3779b97f4a7c15UL;z=state;z=(z^(z>>30))*0xbf58476d1ce4e5b9UL;z=(z^(z>>27))*0x94d049bb133111ebUL;z^=z>>31;}
  double threshold=(double)(z>>11)/9007199254740992.0*sum;
  for(int j=0;j<used;j++){if(threshold<weights[j])return top[j];threshold-=weights[j];}return top[used-1];
 }
 static int[] ReadIds(string p){string raw=File.ReadAllText(p).Trim().Trim('[',']');return Array.ConvertAll(raw.Split(',',StringSplitOptions.RemoveEmptyEntries),s=>Int32.Parse(s.Trim()));}
 public static int ValidateSampler(string directory) {
  int steps=0;foreach(string path in Directory.GetFiles(directory,"*.generated.json")){
   string prefix=path.Substring(0,path.Length-".generated.json".Length);int[] expected=ReadIds(path);ulong rng=42;
   if(!File.Exists(prefix+".step0.logits.f32"))continue;
   for(int step=0;step<expected.Length;step++){byte[] data=File.ReadAllBytes(prefix+".step"+step+".logits.f32");float[] v=new float[data.Length/4];Buffer.BlockCopy(data,0,v,0,data.Length);int actual=Select(v,true,ref rng);if(actual!=expected[step])throw new Exception("Sampler mismatch "+prefix+" step="+step+" actual="+actual+" expected="+expected[step]);steps++;}
  }if(steps==0)throw new Exception("No sampler oracle");Console.WriteLine("Sampler-port PASS steps="+steps);return steps;
 }
 static void Memory(IntPtr gpu,string phase,ulong reserve){UIntPtr free,total;NativeQuality.ggml_backend_dev_memory(gpu,out free,out total);ulong used=Phase9ADL.DedicatedUsage(phase);ulong conservativePhysicalTotal=total.ToUInt64();ulong physicalFree=used>conservativePhysicalTotal?0:conservativePhysicalTotal-used;Console.WriteLine("MEMORY scope=MS-global-dedicated-usage phase="+phase+" used="+used+" conservative_total="+conservativePhysicalTotal+" physical_free="+physicalFree+" WDDM_budget_headroom="+free.ToUInt64()+" reserve="+reserve);if(physicalFree<reserve)throw new Exception("Physical reserve below2GiB+16MiB");if(free.ToUInt64()<256UL*1024*1024)throw new Exception("WDDM budget headroom below256MiB");}
 static void WriteFloats(string p,float[] v){byte[] bytes=new byte[v.Length*4];Buffer.BlockCopy(v,0,bytes,0,bytes.Length);File.WriteAllBytes(p,bytes);}
 static void Decode(IntPtr ctx,IntPtr token,IntPtr flag,int id,bool logits){Marshal.WriteInt32(token,id);Marshal.WriteByte(flag,logits?(byte)1:(byte)0);var b=NativeQuality.llama_batch_get_one(token,1);b.logits=flag;if(NativeQuality.llama_decode(ctx,b)!=0)throw new Exception("Decode failed");}
 static float[] GetLogits(IntPtr ctx,int nv){IntPtr p=NativeQuality.llama_get_logits_ith(ctx,-1);if(p==IntPtr.Zero)throw new Exception("Missing logits");float[] values=new float[nv];Marshal.Copy(p,values,0,nv);return values;}
 static void Request(IntPtr ctx,IntPtr token,IntPtr flag,int nv,int[] ids,bool sampled,int outputs,string prefix) {
  var e2e=Stopwatch.StartNew();llama_memory_clear(llama_get_memory(ctx),true);
  var prefill=Stopwatch.StartNew();for(int t=0;t<ids.Length;t++)Decode(ctx,token,flag,ids[t],t+1==ids.Length);
  float[] values=GetLogits(ctx,nv);double ttft=e2e.Elapsed.TotalSeconds,prefillTime=prefill.Elapsed.TotalSeconds;
  float[] first=values;int[] generated=new int[outputs];ulong rng=42;double compute=0;
  bool capture=Environment.GetEnvironmentVariable("PHASE13_CAPTURE")=="1";var trajectory=new List<float[]>();
  var decode=Stopwatch.StartNew();for(int step=0;step<outputs;step++){
   if(capture)trajectory.Add(values);
   int id=Select(values,sampled,ref rng);generated[step]=id;if(step+1==outputs)break;
   var forward=Stopwatch.StartNew();Decode(ctx,token,flag,id,true);values=GetLogits(ctx,nv);compute+=forward.Elapsed.TotalSeconds;
  }
  double decodeWall=decode.Elapsed.TotalSeconds,wall=e2e.Elapsed.TotalSeconds;
  if(prefix==null)return;
  File.WriteAllText(prefix+".prompt.tokens.json","["+String.Join(",",ids)+"]");File.WriteAllText(prefix+".generated.json","["+String.Join(",",generated)+"]");
  if(capture){for(int t=0;t<trajectory.Count;t++)WriteFloats(prefix+".step"+t+".logits.f32",trajectory[t]);}
  WriteFloats(prefix+".prompt.logits.f32",first);WriteText(NativeQuality.llama_model_get_vocab(_model),new List<int>(generated),prefix+".generated.txt");WriteFloats(prefix+".last.logits.f32",values);
  File.WriteAllText(prefix+".metrics.json","{\"prompt_tokens\":"+ids.Length+",\"output_tokens\":"+outputs+",\"context_capacity\":512,\"sampled\":"+(sampled?"true":"false")+",\"seed\":"+(sampled?"42":"null")+",\"temperature\":0.7,\"top_k\":40,\"ttft_compute_seconds\":"+Num(ttft)+",\"prefill_with_final_head_seconds\":"+Num(prefillTime)+",\"decode_intervals\":"+(outputs-1)+",\"decode_forward_lmhead_seconds\":"+Num(compute)+",\"decode_wall_seconds\":"+Num(decodeWall)+",\"decode_tokens_per_second\":"+Num((outputs-1)/compute)+",\"e2e_seconds\":"+Num(wall)+",\"decode_wall_tokens_per_second\":"+Num((outputs-1)/decodeWall)+"}");
  Console.WriteLine("VULKAN case="+Path.GetFileName(prefix)+" compute_tok_s="+Num((outputs-1)/compute)+" wall_tok_s="+Num((outputs-1)/decodeWall));
 }

 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] [return:MarshalAs(UnmanagedType.I1)] static extern bool llama_vocab_is_eog(IntPtr vocab,int token);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] static extern int llama_token_to_piece(IntPtr vocab,int token,[Out]byte[] buffer,int length,int lstrip,[MarshalAs(UnmanagedType.I1)]bool special);
 static void WriteText(IntPtr vocab,List<int> ids,string path){using(var stream=new MemoryStream()){foreach(int id in ids){byte[] b=new byte[512];int n=llama_token_to_piece(vocab,id,b,b.Length,0,true);if(n<0){b=new byte[-n];n=llama_token_to_piece(vocab,id,b,b.Length,0,true);}if(n<0)throw new Exception("Token piece");stream.Write(b,0,n);}File.WriteAllBytes(path,stream.ToArray());}}
 static void RequestLong(IntPtr ctx,IntPtr vocab,IntPtr token,IntPtr flag,int nv,int[] ids,int[] teacher,int limit,string prefix) {
  llama_memory_clear(llama_get_memory(ctx),true);var time=Stopwatch.StartNew();
  for(int t=0;t<ids.Length;t++)Decode(ctx,token,flag,ids[t],t+1==ids.Length);
  float[] values=GetLogits(ctx,nv);var predictions=new List<int>();var used=new List<int>();ulong rng=42;bool eos=false;
  File.WriteAllText(prefix+".prompt.tokens.json","["+String.Join(",",ids)+"]");
  for(int step=0;step<limit;step++){
   WriteFloats(prefix+".step"+step+".logits.f32",values);int selected=Select(values,false,ref rng);predictions.Add(selected);int id=teacher==null?selected:teacher[step];used.Add(id);
   if(step%64==0||step+1==limit)Console.WriteLine("VULKAN long="+Path.GetFileName(prefix)+" step="+step+" selected="+selected+" used="+id+" elapsed="+Num(time.Elapsed.TotalSeconds));
   if(teacher==null&&llama_vocab_is_eog(vocab,id)){eos=true;break;}if(step+1==limit)break;
   Decode(ctx,token,flag,id,true);values=GetLogits(ctx,nv);
  }
  File.WriteAllText(prefix+".generated.json","["+String.Join(",",predictions)+"]");File.WriteAllText(prefix+".used.json","["+String.Join(",",used)+"]");
  WriteText(vocab,predictions,prefix+".generated.txt");
  File.WriteAllText(prefix+".metrics.json","{\"prompt_tokens\":"+ids.Length+",\"actual_output_tokens\":"+used.Count+",\"limit\":"+limit+",\"context_capacity\":512,\"teacher_forced\":"+(teacher!=null?"true":"false")+",\"eos\":"+(eos?"true":"false")+",\"highest_processed_position\":"+(ids.Length+used.Count-2)+",\"wall_seconds\":"+Num(time.Elapsed.TotalSeconds)+",\"timing_includes_full_logit_file_writes\":true,\"diagnostic_only\":true}");
 }
 static IntPtr _model;
 public static void Run(string runtime,string modelPath,string[] paths,string outDir,int repeats,int warmTokens,int outputs,bool caseWarmup) {
  NativeQuality.Probe();NativeQuality.ggml_backend_load_all_from_path(runtime);NativeQuality.llama_backend_init();IntPtr model=IntPtr.Zero,ctx=IntPtr.Zero;
  IntPtr devices=Marshal.AllocHGlobal(IntPtr.Size*2),token=Marshal.AllocHGlobal(4),flag=Marshal.AllocHGlobal(1);
  try{
   IntPtr gpu=NativeQuality.ggml_backend_dev_by_name("Vulkan0");if(gpu==IntPtr.Zero)throw new Exception("Vulkan0 absent");const ulong reserve=2UL*1024*1024*1024+16UL*1024*1024;
   NativeQuality.Memory(gpu,"admission",13333954560UL+reserve+256UL*1024*1024);
   Marshal.WriteIntPtr(devices,0,gpu);Marshal.WriteIntPtr(devices,IntPtr.Size,IntPtr.Zero);var mp=NativeQuality.llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=99;mp.split_mode=0;mp.load_mtp=false;
   model=NativeQuality.llama_model_load_from_file(modelPath,mp);if(model==IntPtr.Zero)throw new Exception("Model load failed");_model=model;
   var cp=NativeQuality.llama_context_default_params();cp.n_ctx=512;cp.n_batch=1;cp.n_ubatch=1;cp.n_seq_max=1;cp.n_threads=2;cp.n_threads_batch=2;cp.type_k=0;cp.type_v=0;cp.flash_attn_type=0;
   ctx=NativeQuality.llama_init_from_model(model,cp);if(ctx==IntPtr.Zero)throw new Exception("Context failed");Memory(gpu,"context",reserve);
   IntPtr vocab=NativeQuality.llama_model_get_vocab(model);int nv=NativeQuality.llama_vocab_n_tokens(vocab);if(nv!=248320)throw new Exception("Vocabulary mismatch");
   for(int warm=0;warm<warmTokens;warm++)Decode(ctx,token,flag,9419,false);llama_synchronize(ctx);
   foreach(string path in paths){Memory(gpu,"case",reserve);byte[] text=Encoding.UTF8.GetBytes(File.ReadAllText(path,Encoding.UTF8));int n=-NativeQuality.llama_tokenize(vocab,text,text.Length,null,0,false,true);if(n<=0||n+32>512)throw new Exception("Capacity");int[] ids=new int[n];if(NativeQuality.llama_tokenize(vocab,text,text.Length,ids,n,false,true)!=n)throw new Exception("Tokenization");
    if(n+outputs>512)throw new Exception("Capacity");
    if(caseWarmup)Request(ctx,token,flag,nv,ids,false,outputs,null);
    for(int trial=0;trial<repeats;trial++)Request(ctx,token,flag,nv,ids,false,outputs,Path.Combine(outDir,Path.GetFileNameWithoutExtension(path)+".greedy.trial"+trial));
   }
   Memory(gpu,"completed",reserve);
  }finally{if(ctx!=IntPtr.Zero)NativeQuality.llama_free(ctx);if(model!=IntPtr.Zero)NativeQuality.llama_model_free(model);Marshal.FreeHGlobal(devices);Marshal.FreeHGlobal(token);Marshal.FreeHGlobal(flag);NativeQuality.llama_backend_free();}
 }
}
