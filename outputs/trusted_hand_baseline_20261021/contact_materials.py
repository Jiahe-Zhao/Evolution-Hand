"""Apply and audit shape-level materials before physics initialization."""
def configure(stage,root_path,pad_compliance=True):
 from pxr import Usd,UsdShade,UsdPhysics,PhysxSchema
 settings={'pad':(.8,3000. if pad_compliance else 0.,10. if pad_compliance else 0.),'metal':(.2,0.,0.),'object':(.5,0.,0.)}
 materials={}
 for role,(friction,k,d) in settings.items():
  path='/World/BaselineMaterials/'+role;mat=UsdShade.Material.Define(stage,path)
  api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());api.CreateStaticFrictionAttr(friction);api.CreateDynamicFrictionAttr(friction);api.CreateRestitutionAttr(0.)
  phys=PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim());phys.CreateFrictionCombineModeAttr('average');phys.CreateCompliantContactStiffnessAttr(k);phys.CreateCompliantContactDampingAttr(d)
  phys.CreateCompliantContactAccelerationSpringAttr(False)
  materials[role]=mat
 root=stage.GetPrimAtPath(root_path)
 # Instance proxies cannot be authored. Deinstance only this isolated asset.
 for _ in range(3):
  instances=[p for p in Usd.PrimRange(root) if p.IsInstance()]
  if not instances:break
  for p in instances:p.SetInstanceable(False)
 rows=[]
 for prim in Usd.PrimRange(root):
  if prim.GetTypeName() not in ('Mesh','Cube','Capsule','Sphere','Cylinder','Cone'):continue
  owner=prim
  while owner.IsValid() and not owner.HasAPI(UsdPhysics.CollisionAPI):owner=owner.GetParent()
  if not owner.IsValid() or UsdPhysics.CollisionAPI(owner).GetCollisionEnabledAttr().Get() is False:continue
  UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
  name=str(prim.GetPath());role='pad' if 'finger_pad' in name else ('object' if '/Object' in name or '/Support' in name else 'metal')
  UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[role],bindingStrength='strongerThanDescendants',materialPurpose='physics')
  phys=PhysxSchema.PhysxCollisionAPI.Apply(prim);phys.CreateContactOffsetAttr(.0003);phys.CreateRestOffsetAttr(0.)
  material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')[0]
  rows.append(dict(path=name,role=role,material=str(material.GetPath()),static_friction=UsdPhysics.MaterialAPI(material.GetPrim()).GetStaticFrictionAttr().Get(),compliant_stiffness=PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetCompliantContactStiffnessAttr().Get(),compliant_damping=PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetCompliantContactDampingAttr().Get(),contact_offset_m=phys.GetContactOffsetAttr().Get(),rest_offset_m=phys.GetRestOffsetAttr().Get()))
 # The importer creates a merged collision on each collisions Xform.
 # Remove that aggregate after authoring its individual meshes, avoiding duplicate solids.
 for prim in Usd.PrimRange(root):
  if prim.GetTypeName() not in ('Mesh','Cube','Capsule','Sphere','Cylinder','Cone') and prim.HasAPI(UsdPhysics.CollisionAPI):
   prim.RemoveAPI(UsdPhysics.CollisionAPI)
   if prim.HasAPI(UsdPhysics.MeshCollisionAPI):prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
   if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
 return rows
