from pebble_runtime import *
import warnings,traceback
from newton._src.utils.mesh import MeshAdjacency
original=MeshAdjacency.__init__
records=[]
def inspect(self,*args,**kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        original(self,*args,**kwargs)
    bad=[w for w in caught if 'non-manifold edge' in str(w.message)]
    if bad:
        tri=np.array(self.indices).reshape(-1,3)
        edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
        e,c=np.unique(edges,axis=0,return_counts=True)
        rec=dict(warnings=len(bad),triangles=len(tri),vertex_min=int(tri.min()),vertex_max=int(tri.max()),bad_edges=e[c>2].tolist(),incidence=c[c>2].tolist(),duplicate_triangles=int(len(tri)-len(np.unique(np.sort(tri,axis=1),axis=0))),stack=traceback.format_stack(limit=12))
        records.append(rec)
    for w in caught:warnings.warn(str(w.message),w.category)
MeshAdjacency.__init__=inspect
h=PebbleRuntime(side='left',wrist='six_axis',position=(.25,.25,.35),gravity=(0,0,-9.81))
result=dict(records=records,palm_count=h.palm_count,pad_offset=h.pad_offset,bridge_offset=h.bridge_offset,real_count=h.real_count)
(OUT/'warning_probe.json').write_text(json.dumps(result,indent=2))
print('WARNING_PROBE',json.dumps({k:v for k,v in result.items() if k!='records'}), 'bad calls',len(records),flush=True)
