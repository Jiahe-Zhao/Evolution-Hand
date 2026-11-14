"""Explicit calibrated cover profile; task performance remains to be validated."""
import task_materials as legacy

def covered(label):
    return any(x in label for x in ['/skin_', '/visual_palm_lobe_', '/visual_finger_cover_'])
def builder_hook(task):
    def hook(b,w,p,q):
        legacy.builder_hook(task)(b,w,p,q)
        for i,label in enumerate(b.shape_label):
            if covered(label) or '/Object/' in label:
                b.shape_material_ke[i]=10000.;b.shape_material_kd[i]=100.;b.shape_material_mu[i]=.8
                b.shape_gap[i]=.0001;b.shape_margin[i]=0.
                b.custom_attributes['mujoco:solref_mode'].values[i]=0
    return hook

def apply(stage,task):
    # Backend builder is authoritative; do not rebind skin as metal after import.
    return dict(profile='wrap_calibrated_v1',ke=10000,kd=100,mu=.8,gap=.0001,margin=0,solref_mode='FORCE_SPACE',physics_equivalence_verified=False)

def verify_backend(model,task):
    import numpy as np
    arrays={n:getattr(model,'shape_material_'+n).numpy() for n in ['ke','kd','mu']}
    rows=[]
    for i,label in enumerate(model.shape_label):
        if not covered(label):continue
        for k,v in dict(ke=10000,kd=100,mu=.8).items():assert np.isclose(arrays[k][i],v),(label,k)
        rows.append(dict(label=label,**{n:float(v[i]) for n,v in arrays.items()}))
    assert rows,'No cover shapes imported'
    return dict(passed=True,cover_shapes=len(rows),rows=rows)
