"""Source-solid overlap checks at measured articulated poses."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from collision_gate import _origin, _audit_neutral_overlap

class MechanicalCollisionAudit:
    def __init__(self, hand, urdf):
        self.hand, self.path = hand, Path(urdf).resolve()
        self.root = ET.parse(self.path).getroot()
        parents = {j.find('child').get('link'):j for j in self.root.findall('joint')}
        self.meshes, self.mapping = [], {}
        for link in self.root.findall('link'):
            name = link.get('name');body=name;transform=np.eye(4)
            while body not in hand.body_names:
                joint=parents[body]
                if joint.get('type') != 'fixed':raise ValueError(f'Missing moving body {body}')
                transform=_origin(joint.find('origin'))@transform
                body=joint.find('parent').get('link')
            self.mapping[name]=(hand.body_names.index(body),transform)
            for index,collision in enumerate(link.findall('collision')):
                node=collision.find('geometry/mesh')
                if node is None:raise ValueError('Expected explicit mesh collision geometry')
                mesh=trimesh.load(self.path.parent/node.get('filename'),force='mesh',process=True)
                mesh.apply_scale(np.fromstring(node.get('scale','1 1 1'),sep=' '))
                self.meshes.append((name,index,mesh,_origin(collision.find('origin'))))
    def sample(self):
        positions=self.hand.data.body_pos_w[0].cpu().numpy()
        quats=self.hand.data.body_quat_w[0].cpu().numpy();poses={}
        for name,(index,offset) in self.mapping.items():
            matrix=np.eye(4);matrix[:3,:3]=Rotation.from_quat(quats[index][[1,2,3,0]]).as_matrix()
            matrix[:3,3]=positions[index];poses[name]=matrix@offset
        return _audit_neutral_overlap(self.path,self.root,self.meshes,world_poses=poses)
