"""Numerical evidence only; thresholds must be selected for the tested operation."""
import array
import json
import math
import pathlib
import sys

def read(path):
    a=array.array('f');a.frombytes(pathlib.Path(path).read_bytes())
    if sys.byteorder!='little':a.byteswap()
    return a

a,b=read(sys.argv[1]),read(sys.argv[2])
if len(a)!=len(b) or not a or not all(math.isfinite(v) for v in a) or not all(math.isfinite(v) for v in b):
    raise SystemExit('shape or finite-value check failed')
aa=sum(v*v for v in a);bb=sum(v*v for v in b);ab=sum(x*y for x,y in zip(a,b));sq=sum((x-y)**2 for x,y in zip(a,b))
top_a=sorted(range(len(a)),key=a.__getitem__,reverse=True)[:10]
top_b=sorted(range(len(b)),key=b.__getitem__,reverse=True)[:10]
report={'reference':sys.argv[1],'candidate':sys.argv[2],'values':len(a),
        'max_abs':max(abs(x-y) for x,y in zip(a,b)), 'rmse':math.sqrt(sq/len(a)),
        'relative_rmse':math.sqrt(sq/max(aa,1e-30)), 'cosine':ab/math.sqrt(max(aa*bb,1e-30)),
        'top1_equal':top_a[0]==top_b[0], 'reference_top10':top_a,'candidate_top10':top_b,
        'note':'CPU GGML may quantize activations internally; this is not an identical-arithmetic oracle.'}
print(json.dumps(report,indent=2))
if len(sys.argv)>3:pathlib.Path(sys.argv[3]).write_text(json.dumps(report,indent=2))
