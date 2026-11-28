"""Author explicit Newton contact materials; report, never assert PhysX equivalence."""
from pxr import UsdShade,UsdPhysics,Sdf

def builder_hook(task):
    """Apply before finalize: post-instantiation USD edits miss cloned builders."""
    def hook(builder,world,position,rotation):
        for i,label in enumerate(builder.shape_label):
            if '/Hand/' not in label:continue
            pad='finger_pad' in label
            builder.shape_material_mu[i]=.8 if pad or task=='branch' else .2
            if pad and task!='branch':
                builder.shape_material_ke[i]=1500. if task=='grasp' else 3000.
                builder.shape_material_kd[i]=10.
    return hook

def verify_backend(model,task):
    import numpy as np
    arrays={n:getattr(model,'shape_material_'+n).numpy() for n in ['mu','ke','kd']}
    rows=[]
    for i,label in enumerate(model.shape_label):
        if '/Hand/' not in label:continue
        pad='finger_pad' in label;expected=.8 if pad or task=='branch' else .2
        assert np.isclose(arrays['mu'][i],expected),(label,'friction')
        if pad and task!='branch':
            assert np.isclose(arrays['ke'][i],1500. if task=='grasp' else 3000.),(label,'stiffness')
            assert np.isclose(arrays['kd'][i],10.),(label,'damping')
        rows.append(dict(label=label,**{n:float(v[i]) for n,v in arrays.items()}))
    assert rows,'No hand materials verified'
    return dict(passed=True,shapes=len(rows),rows=rows)

def apply(stage,task):
    settings={'pad':(.8,1500. if task=='grasp' else 3000.,10.),'metal':(.2,None,None)}
    if task=='branch':settings={'pad':(.8,None,None),'metal':(.8,None,None)}
    materials={}
    for role,(friction,k,d) in settings.items():
        material=UsdShade.Material.Define(stage,'/World/MigrationMaterials/'+role)
        physics=UsdPhysics.MaterialAPI.Apply(material.GetPrim());physics.CreateStaticFrictionAttr(friction);physics.CreateDynamicFrictionAttr(friction);physics.CreateRestitutionAttr(0.)
        if k is not None:
            material.GetPrim().CreateAttribute('newton:contactStiffness',Sdf.ValueTypeNames.Float).Set(k)
            material.GetPrim().CreateAttribute('newton:contactDamping',Sdf.ValueTypeNames.Float).Set(d)
        materials[role]=material
    counts={role:0 for role in settings}
    for prim in stage.Traverse():
        if '/Hand/' not in str(prim.GetPath()) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        role='pad' if 'finger_pad' in str(prim.GetPath()) else 'metal'
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[role],bindingStrength='strongerThanDescendants',materialPurpose='physics');counts[role]+=1
    if not counts['pad']:raise RuntimeError('No pad collision material assigned')
    return dict(assigned_shapes=counts,parameters=settings,physics_equivalence_verified=False,
        limitation='Matching friction/stiffness inputs does not imply equal PhysX and MuJoCo contact response; solver parameters require calibration')
