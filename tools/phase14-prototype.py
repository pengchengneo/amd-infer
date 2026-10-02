"""Standalone residual/RMS prototype;does not change the engine path."""
import argparse,json,os,pathlib,statistics,subprocess
from run_engine import ROOT, fresh_output, project_gpu_lock, resolve_runtime, sha256, write_json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--runtime-config',type=pathlib.Path);a=p.parse_args();d=fresh_output(a.output);runtime,e,record=resolve_runtime(a.runtime_config);lib=pathlib.Path(e['LD_LIBRARY_PATH'])/'libamdinfer_kernels.so';e['AMD_INFER_EXPLICIT_STREAM']='0';e['AMD_INFER_STABLE_RMS']='1';out=d/'residual-prototype';src=ROOT/'tools/phase14-residual-prototype.cpp';compiler=pathlib.Path(os.environ.get('ROCM_PATH','/opt/rocm-7.2.3'))/'bin/hipcc';d.mkdir(parents=True,exist_ok=False)
 cmd=[str(compiler),'-O3','-std=c++17','--offload-arch=gfx1201','-x','hip',str(src),'-x','none',str(lib),'-Wl,-rpath,'+str(lib.parent),'-o',str(out)]
 record.update({'build_args':cmd,'source_sha256':sha256(src),'library_sha256':sha256(lib),'env':{k:v for k,v in e.items() if k.startswith('AMD_INFER_')},'scope':'Standalone one-block prototype;cached submission/dependency chain;48 ABBAAB trials;512steps/trial;does not establish full-model speed/quality;no engine source modification'})
 write_json(d/'launch.json',record)
 with (d/'build.log').open('w') as f:r=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError((d/'build.log').read_text()[-6000:])
 with project_gpu_lock():
  with (d/'run.log').open('w') as f:r=subprocess.run([str(out)],env=e,stdout=f,stderr=subprocess.STDOUT)
 write_json(d/'exit.json',{'returncode':r.returncode})
 if r.returncode:raise RuntimeError((d/'run.log').read_text()[-6000:])
 report={'patterns':{},'not_engine_headline':True,'all_operator_bits_identical':True,'steps_per_trial':512,'trials':48,'element_results_compared':4*11*5120*2,'caps_lines':[],'resources':[]}
 for line in (d/'run.log').read_text().splitlines():
  if line.startswith('CAPS'):report['caps_lines'].append(line)
  if line.startswith('RESOURCE'):report['resources'].append(line)
  if not line.startswith('RESIDUAL_PROFILE'):continue
  values=dict(v.split('=') for v in line.split()[1:]);arms=report['patterns'].setdefault(values['pattern'],{'control_us':[],'candidate_us':[]});arms['candidate_us' if values['candidate']=='1' else 'control_us'].append(float(values['chain_us']))
 for arms in report['patterns'].values():
  arms['control_median_us']=statistics.median(arms['control_us'][1:]);arms['candidate_median_us']=statistics.median(arms['candidate_us']);arms['exclude_first_control_by_fixed_rule']=True;arms['saved_us']=arms['control_median_us']-arms['candidate_median_us'];arms['extrapolated_64_layer_saved_ms']=arms['saved_us']*64/1000
 write_json(d/'summary.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
