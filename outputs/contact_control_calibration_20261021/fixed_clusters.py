"""Preserve production collision semantics when fixed links are not merged for sensors."""
import xml.etree.ElementTree as ET
import importlib.util

def exclusions(path):
 spec=importlib.util.spec_from_file_location('contact_topology','/home/zjh/Evolution_PC/evolution_tasks/collision_topology.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 root=ET.parse(path).getroot();names=[x.get('name') for x in root.findall('link')]
 parents={j.find('child').get('link'):(j.find('parent').get('link'),j.get('type')) for j in root.findall('joint')}
 def representative(n):
  while n in parents and parents[n][1]=='fixed':n=parents[n][0]
  return n
 reps={n:representative(n) for n in names};production=set(module.collision_exclusions(path,set(reps.values())))
 return [(a,b) for i,a in enumerate(names) for b in names[i+1:] if reps[a]==reps[b] or tuple(sorted((reps[a],reps[b]))) in production]

def apply(stage,path,root_path):
 from pxr import Usd,UsdPhysics
 bodies={p.GetName():p for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
 pairs=exclusions(path)
 for a,b in pairs:
  if a in bodies and b in bodies:UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
 return pairs
