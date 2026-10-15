"""Add a load-bearing thumb yoke without changing any joint coordinates."""
import argparse,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from audit_mechanical_workspace import graph,fk
from compile_mechanical_hand import origin


def append_parts(link,parts,mesh_dir):
    old=link.find('inertial');T=origin(old.find('origin'));mass=float(old.find('mass').get('value'));n=old.find('inertia')
    I=np.array([[float(n.get('ixx')),float(n.get('ixy')),float(n.get('ixz'))],[float(n.get('ixy')),float(n.get('iyy')),float(n.get('iyz'))],[float(n.get('ixz')),float(n.get('iyz')),float(n.get('izz'))]])
    values=[(mass,T[:3,3],T[:3,:3]@I@T[:3,:3].T)]
    for name,mesh in parts:
        mesh.density=2700.;values.append((mesh.mass,mesh.center_mass,mesh.moment_inertia))
        path=mesh_dir/f'{link.get("name")}_{name}.stl';mesh.export(path)
        for kind in ('visual','collision'):
            e=ET.SubElement(link,kind,name=name);ET.SubElement(e,'origin',xyz='0 0 0',rpy='0 0 0')
            g=ET.SubElement(e,'geometry');ET.SubElement(g,'mesh',filename='../meshes/'+path.name)
            if kind=='visual':
                m=ET.SubElement(e,'material',name=name);ET.SubElement(m,'color',rgba='0.3 0.38 0.43 1')
    total=sum(v[0] for v in values);center=sum(m*c for m,c,_ in values)/total;tensor=np.zeros((3,3))
    for m,c,I in values:
        delta=c-center;tensor+=I+m*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
    link.remove(old);e=ET.SubElement(link,'inertial');ET.SubElement(e,'origin',xyz=' '.join(map(str,center)),rpy='0 0 0');ET.SubElement(e,'mass',value=str(total))
    ET.SubElement(e,'inertia',**{k:str(tensor[i,j]) for k,i,j in [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]})
    return total-mass


def main(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve();tree=ET.parse(source);root=tree.getroot()
    mesh_dir=destination.parent.parent/'meshes';mesh_dir.mkdir(parents=True,exist_ok=True);destination.parent.mkdir(parents=True,exist_ok=True)
    for m in root.findall('.//mesh'):
        original=(source.parent/m.get('filename')).resolve();shutil.copy2(original,mesh_dir/original.name);m.set('filename','../meshes/'+original.name)
    nodes,base=graph(source);poses=fk(nodes,base,{j.get('name'):0. for j in nodes if j.get('type')!='fixed'})
    joint=root.find("joint[@name='link_1_thumb_spread_joint']");child=joint.find('child').get('link');center=poses[child][:3,3];axis=poses[child][:3,:3]@np.fromstring(joint.find('axis').get('xyz'),sep=' ')
    parts=[]
    for side in (-1,1):
        end=center+side*.007*axis;start=side*.007*axis
        beam=trimesh.creation.box(extents=[.003,.003,np.linalg.norm(end-start)])
        beam.apply_transform(trimesh.geometry.align_vectors([0,0,1],end-start));beam.apply_translation((start+end)/2)
        cheek=trimesh.creation.cylinder(radius=.004,height=.002,sections=24)
        cheek.apply_transform(trimesh.geometry.align_vectors([0,0,1],axis));cheek.apply_translation(end)
        parts.extend([(f'thumb_brace_{side+1}',beam),(f'thumb_cheek_{side+1}',cheek)])
    added=append_parts(root.find(f"link[@name='{base}']"),parts,mesh_dir)
    shaft=trimesh.creation.cylinder(radius=.0018,height=.016,sections=24)
    shaft.apply_transform(trimesh.geometry.align_vectors([0,0,1],np.fromstring(joint.find('axis').get('xyz'),sep=' ')))
    added+=append_parts(root.find(f"link[@name='{child}']"),[('opposition_shaft',shaft)],mesh_dir)
    ET.indent(tree,space='  ');tree.write(destination,encoding='unicode',xml_declaration=True)
    report={'source':str(source),'joint_coordinates_unchanged':True,'added_mass_kg':added,'thumb_axis_center_m':center.tolist(),'thumb_axis_direction':axis.tolist(),'yoke_inner_gap_m':.012,'shaft_diameter_m':.0036,'status':'candidate_requires_collision_and_motion_validation'}
    destination.with_suffix('.thumb_mount.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();main(a.source,a.destination)
