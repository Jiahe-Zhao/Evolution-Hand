"""Independent USD wrapper: original Shadow finger geometry/tendons untouched."""
from pathlib import Path
import json
NAMES=['diagnostic_wrist_'+a for a in 'xyz']

def build(source,destination):
 from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf
 destination=Path(destination)
 if destination.exists():raise FileExistsError(destination)
 original=Usd.Stage.Open(str(source));stage=Usd.Stage.CreateNew(str(destination))
 root=original.GetDefaultPrim().GetPath();p=stage.DefinePrim(root,'Xform');p.GetReferences().AddReference(str(source),root);stage.SetDefaultPrim(p)
 UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
 fixed=UsdPhysics.FixedJoint(stage.GetPrimAtPath(root.AppendPath('joints/rootJoint')))
 mount=fixed.GetBody1Rel().GetTargets()[0]
 # All stages are colocated with the original mounting frame at zero position.
 cache=UsdGeom.XformCache();matrix=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(mount))*cache.GetLocalToWorldTransform(p).GetInverse()
 bodies=[]
 for name in ['diagnostic_world','diagnostic_stage_x','diagnostic_stage_y']:
  path=root.AppendChild(name);body=UsdGeom.Xform.Define(stage,path);body.AddTransformOp().Set(matrix)
  UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());m=UsdPhysics.MassAPI.Apply(body.GetPrim());m.CreateMassAttr(.05);m.CreateDiagonalInertiaAttr(Gf.Vec3f(.00001));m.CreateCenterOfMassAttr(Gf.Vec3f(0));bodies.append(path)
 fixed.GetBody1Rel().SetTargets([bodies[0]])
 for i,name in enumerate(NAMES):
  j=UsdPhysics.PrismaticJoint.Define(stage,root.AppendPath('joints/'+name));j.CreateBody0Rel().SetTargets([bodies[i]]);j.CreateBody1Rel().SetTargets([bodies[i+1] if i<2 else mount]);j.CreateAxisAttr('XYZ'[i]);j.CreateLowerLimitAttr(-.25);j.CreateUpperLimitAttr(.25)
  j.CreateLocalPos0Attr(Gf.Vec3f(0));j.CreateLocalPos1Attr(Gf.Vec3f(0));j.CreateLocalRot0Attr(Gf.Quatf(1));j.CreateLocalRot1Attr(Gf.Quatf(1))
  d=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');d.CreateTypeAttr('force');d.CreateStiffnessAttr(3000.);d.CreateDampingAttr(100.);d.CreateMaxForceAttr(100.)
  PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateMaxJointVelocityAttr(.02)
 stage.GetRootLayer().Save()
 # Check every original authored attribute/relationship, excluding the one reparented root joint.
 changed=[]
 for prim in original.Traverse():
  if prim.GetPath()==fixed.GetPath():continue
  new=stage.GetPrimAtPath(prim.GetPath())
  for a in prim.GetAttributes():
   if str(a.Get())!=str(new.GetAttribute(a.GetName()).Get()):changed.append(str(a.GetPath()))
  for rel in prim.GetRelationships():
   if rel.GetTargets()!=new.GetRelationship(rel.GetName()).GetTargets():changed.append(str(rel.GetPath()))
 if changed:raise RuntimeError('Unexpected stock property differences: '+str(changed[:20]))
 report=dict(source=str(source),wrapper=str(destination),original_attributes_relationships_unchanged=True,only_reparented_joint=str(fixed.GetPath()),hand_root=str(mount),wrist_joint_names=NAMES,added_moving_mass_kg=.1,position_limits_m=[-.25,.25],speed_limit_m_s=.02,force_limit_N=100,stiffness_N_m=3000,damping_N_s_m=100,axes_frame='stock mounting frame')
 destination.with_suffix('.json').write_text(json.dumps(report,indent=2));return report
