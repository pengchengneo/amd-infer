// Independent pinned llama.cpp b11284 public C API. No AMDInfer graph/math.
using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
public static class NativeQuality {
 [StructLayout(LayoutKind.Sequential)] public struct ModelParams {
  public IntPtr devices, tensor_buft_overrides;
  public int n_gpu_layers, split_mode, load_mode, lazy_mode, main_gpu;
  public IntPtr tensor_split, progress_callback, progress_callback_user_data, kv_overrides;
  [MarshalAs(UnmanagedType.I1)] public bool vocab_only;
  [MarshalAs(UnmanagedType.I1)] public bool check_tensors;
  [MarshalAs(UnmanagedType.I1)] public bool use_extra_bufts;
  [MarshalAs(UnmanagedType.I1)] public bool no_host;
  [MarshalAs(UnmanagedType.I1)] public bool no_alloc;
  [MarshalAs(UnmanagedType.I1)] public bool load_mtp;
 }
 [StructLayout(LayoutKind.Sequential)] public struct ContextParams {
  public uint n_ctx,n_batch,n_ubatch,n_seq_max,n_rs_seq,n_outputs_max,n_outputs_max_per_seq;
  public int n_threads,n_threads_batch,ctx_type,rope_scaling_type,pooling_type,attention_type,flash_attn_type;
  public float rope_freq_base,rope_freq_scale,yarn_ext_factor,yarn_attn_factor,yarn_beta_fast,yarn_beta_slow;
  public uint yarn_orig_ctx; public float defrag_thold;
  public IntPtr cb_eval,cb_eval_user_data;
  public int type_k,type_v;
  public IntPtr abort_callback,abort_callback_data;
  [MarshalAs(UnmanagedType.I1)] public bool embeddings;
  [MarshalAs(UnmanagedType.I1)] public bool offload_kqv;
  [MarshalAs(UnmanagedType.I1)] public bool no_perf;
  [MarshalAs(UnmanagedType.I1)] public bool op_offload;
  [MarshalAs(UnmanagedType.I1)] public bool swa_full;
  [MarshalAs(UnmanagedType.I1)] public bool kv_unified;
  public IntPtr samplers; public UIntPtr n_samplers; public IntPtr ctx_other;
 }
 [StructLayout(LayoutKind.Sequential)] public struct Batch {
  public int n_tokens; public IntPtr token,embd,pos,n_seq_id,seq_id,logits;
 }
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
 public static extern bool SetDllDirectory(string path);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern ModelParams llama_model_default_params();
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern ContextParams llama_context_default_params();
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern void llama_backend_init();
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern void llama_backend_free();
 [DllImport("ggml.dll",CallingConvention=CallingConvention.Cdecl,CharSet=CharSet.Ansi)] public static extern void ggml_backend_load_all_from_path(string path);
 [DllImport("ggml.dll",CallingConvention=CallingConvention.Cdecl,CharSet=CharSet.Ansi)] public static extern IntPtr ggml_backend_dev_by_name(string name);
 [DllImport("ggml-base.dll",CallingConvention=CallingConvention.Cdecl)] public static extern void ggml_backend_dev_memory(IntPtr dev,out UIntPtr free,out UIntPtr total);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl,CharSet=CharSet.Ansi)] public static extern IntPtr llama_model_load_from_file(string path,ModelParams param);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr llama_init_from_model(IntPtr model,ContextParams param);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern void llama_model_free(IntPtr model);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern void llama_free(IntPtr context);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr llama_model_get_vocab(IntPtr model);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern int llama_vocab_n_tokens(IntPtr vocab);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern int llama_tokenize(IntPtr vocab,byte[] text,int len,[Out] int[] tokens,int max,[MarshalAs(UnmanagedType.I1)]bool add,[MarshalAs(UnmanagedType.I1)]bool special);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern Batch llama_batch_get_one(IntPtr tokens,int count);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern int llama_decode(IntPtr context,Batch batch);
 [DllImport("llama.dll",CallingConvention=CallingConvention.Cdecl)] public static extern IntPtr llama_get_logits_ith(IntPtr context,int index);
 public static void Probe() {
  var m=llama_model_default_params();var c=llama_context_default_params();
  if(Marshal.SizeOf(typeof(ModelParams))!=80 || Marshal.SizeOf(typeof(ContextParams))!=160 || Marshal.SizeOf(typeof(Batch))!=56)throw new Exception("Pinned ABI sizes mismatch");
  if(c.n_batch!=2048 || c.n_ubatch!=512 || c.n_ctx!=512 || c.type_k!=1 || c.type_v!=1 || c.cb_eval!=IntPtr.Zero || m.kv_overrides!=IntPtr.Zero)throw new Exception("Pinned default parameter probe mismatch; do not load model");
  Console.WriteLine("ABI probe sizes=80/160/56 default_ctx="+c.n_ctx+" batch="+c.n_batch+" ubatch="+c.n_ubatch+" gpu_layers="+m.n_gpu_layers);
 }
 public static void Memory(IntPtr dev,string phase,ulong reserve) {
  UIntPtr f,t;ggml_backend_dev_memory(dev,out f,out t);Console.WriteLine("NATIVE memory phase="+phase+" free="+f.ToUInt64()+" total="+t.ToUInt64());
  if(f.ToUInt64()<reserve)throw new Exception("Reference free memory below reserve; abort own test");
 }
 public static void Run(string runtime,string modelPath,string[] paths,string outDir,int context,int steps) {
  Probe();ggml_backend_load_all_from_path(runtime);llama_backend_init();
  IntPtr devices=Marshal.AllocHGlobal(2*IntPtr.Size),model=IntPtr.Zero;
  try {
   IntPtr gpu=ggml_backend_dev_by_name("Vulkan0");if(gpu==IntPtr.Zero)throw new Exception("Vulkan0 absent");
   const ulong reserve=2UL*1024*1024*1024+16UL*1024*1024;
   Memory(gpu,"admission",13333954560UL+reserve+256UL*1024*1024);
   Marshal.WriteIntPtr(devices,0,gpu);Marshal.WriteIntPtr(devices,IntPtr.Size,IntPtr.Zero);
   var mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=99;mp.split_mode=0;mp.load_mtp=false;
   model=llama_model_load_from_file(modelPath,mp);if(model==IntPtr.Zero)throw new Exception("Model load failed");Memory(gpu,"model",reserve);
   IntPtr vocab=llama_model_get_vocab(model);int nv=llama_vocab_n_tokens(vocab);if(nv!=248320)throw new Exception("Unexpected vocabulary");
   foreach(string path in paths) {
    byte[] text=Encoding.UTF8.GetBytes(File.ReadAllText(path,Encoding.UTF8));int n=-llama_tokenize(vocab,text,text.Length,null,0,false,true);
    if(n<=0 || n+steps>context)throw new Exception("Prompt/context capacity");int[] ids=new int[n];if(llama_tokenize(vocab,text,text.Length,ids,n,false,true)!=n)throw new Exception("Tokenize failed");
    string prefix=Path.Combine(outDir,Path.GetFileNameWithoutExtension(path));File.WriteAllText(prefix+".tokens.json","["+String.Join(",",ids)+"]",new UTF8Encoding(false));
    var cp=llama_context_default_params();cp.n_ctx=(uint)context;cp.n_batch=1;cp.n_ubatch=1;cp.n_seq_max=1;cp.n_threads=2;cp.n_threads_batch=2;cp.type_k=0;cp.type_v=0;cp.flash_attn_type=0;
    IntPtr ctx=llama_init_from_model(model,cp),token=Marshal.AllocHGlobal(4);if(ctx==IntPtr.Zero)throw new Exception("Context creation failed");
    try {
     Memory(gpu,"context",reserve);foreach(int id in ids){Marshal.WriteInt32(token,id);if(llama_decode(ctx,llama_batch_get_one(token,1))!=0)throw new Exception("Prefill decode failed");}
     int[] generated=new int[steps];float[] values=new float[nv];byte[] data=new byte[nv*4];
     for(int step=0;step<steps;step++) {
      IntPtr p=llama_get_logits_ith(ctx,-1);if(p==IntPtr.Zero)throw new Exception("Missing logits");Marshal.Copy(p,values,0,nv);Buffer.BlockCopy(values,0,data,0,data.Length);File.WriteAllBytes(prefix+".step"+step+".logits.f32",data);
      int best=0;for(int i=0;i<nv;i++){if(Single.IsNaN(values[i])||Single.IsInfinity(values[i]))throw new Exception("Nonfinite reference");if(values[i]>values[best])best=i;}generated[step]=best;
      if(step+1<steps){Marshal.WriteInt32(token,best);if(llama_decode(ctx,llama_batch_get_one(token,1))!=0)throw new Exception("Generation decode failed");}
     }
     File.WriteAllText(prefix+".generated.json","["+String.Join(",",generated)+"]",new UTF8Encoding(false));
     Console.WriteLine("REFERENCE case="+Path.GetFileNameWithoutExtension(path)+" prompt="+n+" sampled_steps="+steps+" processed_positions="+(n+steps-1)+" capacity="+context);
    }finally{llama_free(ctx);Marshal.FreeHGlobal(token);}
   }
  }finally{if(model!=IntPtr.Zero)llama_model_free(model);Marshal.FreeHGlobal(devices);llama_backend_free();}
 }
}
