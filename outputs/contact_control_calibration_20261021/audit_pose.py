import sys,json,importlib.util
from pathlib import Path
import fcl,numpy as np
from plan_grasp import Model
spec=importlib.util.spec_from_file_location('topology','/home/zjh/Evolution_PC/evolution_tasks/collision_topology.py');top=importlib.util.module_from_spec(spec);spec.loader.exec_module(top)
m=Model(sys.argv[1]);excluded=set(tuple(sorted(p)) for p in __import__("fixed_clusters").exclusions(sys.argv[1]))
results=[]
for row in json.loads(Path(sys.argv[2]).read_text()):
 q=[row['q'][n] for n in m.names];ts=m.fk(q);objects=[]
 for name,part,shape in m.meshes:
  T=ts[name];objects.append((name,part,fcl.CollisionObject(shape,fcl.Transform(T[:3,:3],T[:3,3]))))
 hits=[]
 for i,(a,pa,oa) in enumerate(objects):
  for b,pb,ob in objects[i+1:]:
   if a==b or tuple(sorted((a,b))) in excluded:continue
   res=fcl.CollisionResult();fcl.collide(oa,ob,fcl.CollisionRequest(enable_contact=True,num_max_contacts=3),res)
   if res.is_collision:hits.append({'a':a,'b':b,'parts':[pa,pb],'depth_mm':1000*max((c.penetration_depth for c in res.contacts),default=0.)})
 row['self_intersections']=hits;results.append(row)
 print('COST',row['cost'],'OBJECT_HITS',len(row['intersections']),'SELF_HITS',len(hits),'MAX_MM',max((h['depth_mm'] for h in hits),default=0.),'TOP',sorted(hits,key=lambda h:-h['depth_mm'])[:6],flush=True)
Path(sys.argv[3]).write_text(json.dumps(results,indent=2))
