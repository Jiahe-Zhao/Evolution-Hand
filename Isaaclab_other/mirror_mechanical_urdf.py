"""Reflect a complete mechanical asset across X=0, retaining joint coordinates."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def mirror(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    root.set('name', root.get('name') + '_left')
    reflection = np.diag([-1., 1., 1.])
    meshes = destination.parent.parent / 'meshes'
    meshes.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted = {}
    for mesh_node in root.findall('.//mesh'):
        original = (source.parent / mesh_node.get('filename')).resolve()
        if original not in converted:
            output = meshes / f'mirror_{len(converted):03d}_{original.name}'
            mesh = trimesh.load(original, force='mesh')
            transform = np.eye(4); transform[:3, :3] = reflection
            mesh.apply_transform(transform)
            mesh.export(output)
            converted[original] = output
        mesh_node.set('filename', '../meshes/' + converted[original].name)
    for origin in root.findall('.//origin'):
        xyz = reflection @ np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rotation = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        rpy = Rotation.from_matrix(reflection @ rotation @ reflection).as_euler('xyz')
        origin.set('xyz', ' '.join(map(str, xyz)))
        origin.set('rpy', ' '.join(map(str, rpy)))
    for axis in root.findall('joint/axis'):
        # Angular axes are pseudovectors: det(M) * M * axis.
        vector = -reflection @ np.fromstring(axis.get('xyz'), sep=' ')
        axis.set('xyz', ' '.join(map(str, vector)))
    for inertia in root.findall('link/inertial/inertia'):
        for key in ('ixy', 'ixz'):
            inertia.set(key, str(-float(inertia.get(key))))
    ET.indent(tree, space='  ')
    tree.write(destination, encoding='unicode', xml_declaration=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('destination')
    args = parser.parse_args()
    mirror(args.source, args.destination)
