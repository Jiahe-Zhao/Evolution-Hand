"""Build a continuous tetrahedral palm layer from the four proximal support rays.

Geometry/material/attachment proposal only. Runtime deformable coupling must be
verified before using this layer in task scores or calling it a flexible hand.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from audit_mechanical_workspace import graph, fk


def build(urdf, output):
    nodes, base = graph(urdf)
    poses = fk(nodes, base, {j.get('name'): 0. for j in nodes if j.get('type') != 'fixed'})
    roots = np.array([poses[f'link_{d}_0'][:3, 3] for d in range(2, 6)])
    ends = np.array([poses[f'link_{d}_1'][:3, 3] for d in range(2, 6)])
    # Cover the proximal support region, leaving the distal hinge clear.
    nu, nv, nw = 19, 13, 3
    points = []
    for u in np.linspace(0, 3, nu):
        digit = min(int(u), 2); weight = u - digit
        root = (1-weight)*roots[digit] + weight*roots[digit+1]
        end = (1-weight)*ends[digit] + weight*ends[digit+1]
        transverse = (roots[digit+1] + ends[digit+1] - roots[digit] - ends[digit]) / 2
        normal = np.cross(end-root, transverse)
        normal /= np.linalg.norm(normal)
        for v in np.linspace(0.05, 0.80, nv):
            for w in np.linspace(-0.0015, 0.0015, nw):
                points.append(root + v*(end-root) + w*normal)
    points = np.asarray(points)
    def idx(i,j,k): return (i*nv+j)*nw+k
    tets = []
    for i in range(nu-1):
        for j in range(nv-1):
            for k in range(nw-1):
                c=[idx(i,j,k),idx(i+1,j,k),idx(i,j+1,k),idx(i+1,j+1,k),idx(i,j,k+1),idx(i+1,j,k+1),idx(i,j+1,k+1),idx(i+1,j+1,k+1)]
                tets.extend([[c[a] for a in t] for t in [(0,1,3,7),(0,3,2,7),(0,2,6,7),(0,6,4,7),(0,4,5,7),(0,5,1,7)]])
    tets=np.asarray(tets)
    q=points[tets]
    volumes=np.linalg.det(np.stack([q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]],axis=-1))/6
    assert np.abs(volumes).min()>1e-15, 'Degenerate palm tetrahedron'
    negative=volumes<0
    tets[negative,0],tets[negative,1]=tets[negative,1].copy(),tets[negative,0].copy()
    faces=np.concatenate([tets[:,[0,1,2]],tets[:,[0,1,3]],tets[:,[0,2,3]],tets[:,[1,2,3]]])
    _,index,count=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    surface=trimesh.Trimesh(vertices=points,faces=faces[index[count==1]],process=True)
    surface.fix_normals()
    assert surface.is_watertight and len(surface.split())==1
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'palm_tetrahedra.npz',points=points,tetrahedra=tets)
    surface.export(output/'continuous_palm.stl')
    anchors={}
    for digit,i in zip(range(2,6),[0,6,12,18]):
        ids=[idx(i,j,0) for j in range(nv)]
        inverse=np.linalg.inv(poses[f'link_{digit}_0'])
        local=(inverse[:3,:3]@points[ids].T).T+inverse[:3,3]
        anchors[f'link_{digit}_0']={'node_indices':ids,'local_points_m':local.tolist()}
    report={'status':'geometry_ready_runtime_unvalidated','actuated_joint_count_unchanged':20,
            'thickness_m':0.003,'vertices':len(points),'tetrahedra':len(tets),
            'watertight':bool(surface.is_watertight),'connected_components':1,
            'material_design_assumptions':{'youngs_modulus_pa':300000,'poissons_ratio':0.45,'density_kg_m3':1100},
            'volume_m3':float(abs(volumes).sum()),'attachment_proposal':anchors,
            'coverage':'5% to 80% of four palm-root support segments; thumb remains independently articulated',
            'pending':['Verify root-ray geometry corresponds to intended palm region','Couple to articulated links in PhysX','Check strain, self-contact, object contact and full task performance']}
    (output/'compliant_palm.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='attachment_proposal'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('urdf');p.add_argument('output');a=p.parse_args();build(a.urdf,a.output)
