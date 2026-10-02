// Project-authored read-only ADL ABI client. No settings, logging-start or setters.
using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
public static class Phase9ADL {
 [UnmanagedFunctionPointer(CallingConvention.Cdecl)] delegate IntPtr Alloc(int size);
 [DllImport("atiadlxx.dll",CallingConvention=CallingConvention.Cdecl)] static extern int ADL2_Main_Control_Create(Alloc malloc,int connected,out IntPtr ctx);
 [DllImport("atiadlxx.dll",CallingConvention=CallingConvention.Cdecl)] static extern int ADL2_Main_Control_Destroy(IntPtr ctx);
 [DllImport("atiadlxx.dll",CallingConvention=CallingConvention.Cdecl)] static extern int ADL2_Adapter_NumberOfAdapters_Get(IntPtr ctx,out int count);
 [DllImport("atiadlxx.dll",CallingConvention=CallingConvention.Cdecl)] static extern int ADL2_New_QueryPMLogData_Get(IntPtr ctx,int index,IntPtr data);
 [DllImport("atiadlxx.dll",CallingConvention=CallingConvention.Cdecl)] static extern int ADL2_Adapter_AdapterInfo_Get(IntPtr ctx,IntPtr info,int size);
 [DllImport("atiadlxx.dll",CallingConvention=CallingConvention.Cdecl)] static extern int ADL2_Adapter_DedicatedVRAMUsage_Get(IntPtr ctx,int index,out int usedMiB);
 public static ulong DedicatedUsage(string prefix) {
  var allocations=new List<IntPtr>();Alloc callback=(n)=>{var p=Marshal.AllocHGlobal(n);allocations.Add(p);return p;};IntPtr ctx=IntPtr.Zero;ulong result=0;bool found=false;
  try {int rc=ADL2_Main_Control_Create(callback,1,out ctx);if(rc!=0)throw new Exception("ADL create");int count;rc=ADL2_Adapter_NumberOfAdapters_Get(ctx,out count);if(rc!=0||count<1||count>16)throw new Exception("ADL count");
   IntPtr info=Marshal.AllocHGlobal(count*1572);try{for(int k=0;k<count*1572/4;k++)Marshal.WriteInt32(info,k*4,0);for(int k=0;k<count;k++)Marshal.WriteInt32(info,k*1572,1572);
    rc=ADL2_Adapter_AdapterInfo_Get(ctx,info,count*1572);if(rc!=0)throw new Exception("ADL info rc="+rc);
    for(int k=0;k<count;k++){int index=Marshal.ReadInt32(info,k*1572+4);string name=Marshal.PtrToStringAnsi(IntPtr.Add(info,k*1572+280),256).TrimEnd('\0');int used;rc=ADL2_Adapter_DedicatedVRAMUsage_Get(ctx,index,out used);
     Console.WriteLine("ADL physical phase="+prefix+" index="+index+" name="+name+" rc="+rc+" usedMiB="+used);
     if(name.Contains("9070")&&rc==0&&used>=0&&used<=16384){result=Math.Max(result,(ulong)used*1024*1024);found=true;}
    }
   }finally{Marshal.FreeHGlobal(info);}
  }finally{if(ctx!=IntPtr.Zero)ADL2_Main_Control_Destroy(ctx);foreach(var p in allocations)Marshal.FreeHGlobal(p);GC.KeepAlive(callback);}if(!found)throw new Exception("9070 physical memory counter unavailable");return result;
 }
 public static void Run(string output,int samples) {
  var allocations=new List<IntPtr>(); Alloc callback=(n)=>{var p=Marshal.AllocHGlobal(n);allocations.Add(p);return p;};IntPtr ctx=IntPtr.Zero;
  using(var w=new StreamWriter(output,false,new System.Text.UTF8Encoding(false))) {
   try {int rc=ADL2_Main_Control_Create(callback,1,out ctx);w.WriteLine("{\"create_rc\":"+rc+"}");w.Flush();if(rc!=0)return;
    int count;rc=ADL2_Adapter_NumberOfAdapters_Get(ctx,out count);w.WriteLine("{\"adapters_rc\":"+rc+",\"count\":"+count+"}");if(rc!=0||count<1||count>16)return;
    var p=Marshal.AllocHGlobal(4+256*8);try {
     for(int t=0;t<samples;t++){for(int index=0;index<count;index++){
      for(int k=0;k<513;k++)Marshal.WriteInt32(p,k*4,0);Marshal.WriteInt32(p,2052);
      rc=ADL2_New_QueryPMLogData_Get(ctx,index,p);var values=new List<string>();
      if(rc==0)for(int sensor=0;sensor<256;sensor++)if(Marshal.ReadInt32(p,4+sensor*8)!=0)values.Add("\""+sensor+"\":"+Marshal.ReadInt32(p,8+sensor*8));
      w.WriteLine("{\"utc\":\""+DateTime.UtcNow.ToString("o")+"\",\"adapter\":"+index+",\"rc\":"+rc+",\"supported_values\":{"+String.Join(",",values)+"}}");
     }w.Flush();if(t+1<samples)Thread.Sleep(1000);}
    }finally{Marshal.FreeHGlobal(p);}
   }finally{if(ctx!=IntPtr.Zero)ADL2_Main_Control_Destroy(ctx);foreach(var p in allocations)Marshal.FreeHGlobal(p);GC.KeepAlive(callback);}
  }
 }
}
