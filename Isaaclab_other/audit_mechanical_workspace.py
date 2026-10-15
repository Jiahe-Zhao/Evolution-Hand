"""Sample FK workspace and verify equal-coordinate left/right reflection."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def graph(path):
    root = ET.parse(path).getroot()
    nodes = root.findall('joint')
    links = {x.get('name') for x in root.findall('link')}
    children = {j.find('child').get('link') for j in nodes}
    base, = links - children
    return nodes, base


def fk(nodes, base, angles):
    poses = {base: np.eye(4)}
    remaining = list(nodes)
    while remaining:
        progressed = False
        for joint in remaining[:]:
            parent = joint.find('parent').get('link')
            if parent not in poses:
                continue
            origin = joint.find('origin')
            local = np.eye(4)
            if origin is not None:
                local[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                local[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            if joint.get('type') != 'fixed':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * angles[joint.get('name')]).as_matrix()
                local = local @ motion
            poses[joint.find('child').get('link')] = poses[parent] @ local
            remaining.remove(joint); progressed = True
        if not progressed:
            raise ValueError('Disconnected or cyclic joint graph')
    return poses


def main(right, left, output):
    right_nodes, right_root = graph(right)
    left_nodes, left_root = graph(left)
    moving = [j for j in right_nodes if j.get('type') != 'fixed']
    left_by_name = {j.get('name'): j for j in left_nodes}
    for j in moving:
        for bound in ('lower', 'upper'):
            assert float(j.find('limit').get(bound)) == float(left_by_name[j.get('name')].find('limit').get(bound))
    reflection = np.diag([-1., 1., 1., 1.])
    rng = np.random.default_rng(42)
    peak = 0.
    endpoints = {}
    collision_free_endpoints = {}
    collision_results = []
    from collision_gate import _audit_neutral_overlap, _origin
    xml_root = ET.parse(right).getroot()
    local_meshes = []
    for link in xml_root.findall('link'):
        for index, collision in enumerate(link.findall('collision')):
            node = collision.find('geometry/mesh')
            mesh = trimesh.load(Path(right).parent / node.get('filename'), force='mesh', process=True)
            mesh.apply_scale(np.fromstring(node.get('scale', '1 1 1'), sep=' '))
            local_meshes.append((link.get('name'), index, mesh, _origin(collision.find('origin'))))
    parents = {j.find('parent').get('link') for j in right_nodes}
    leaves = [j.find('child').get('link') for j in right_nodes if j.find('child').get('link') not in parents]
    offsets = {}
    links = {n.get('name'): n for n in ET.parse(right).getroot().findall('link')}
    for leaf in leaves:
        surfaces = []
        for collision in links[leaf].findall('collision'):
            node = collision.find('geometry/mesh')
            mesh = trimesh.load(Path(right).parent / node.get('filename'), force='mesh')
            vertices = np.asarray(mesh.vertices) * np.fromstring(node.get('scale', '1 1 1'), sep=' ')
            origin = collision.find('origin')
            if origin is not None:
                vertices = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).apply(vertices) + np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            surfaces.append(vertices)
        vertices = np.concatenate(surfaces)
        offsets[leaf] = vertices[vertices[:, 2] >= vertices[:, 2].max() - 1e-6].mean(axis=0)
    for sample in range(1001):
        angles = {j.get('name'): (0. if sample == 0 else rng.uniform(float(j.find('limit').get('lower')), float(j.find('limit').get('upper')))) for j in moving}
        r, l = fk(right_nodes, right_root, angles), fk(left_nodes, left_root, angles)
        for link in r:
            peak = max(peak, float(np.abs(l[link] - reflection @ r[link] @ reflection).max()))
        collision_free = False
        if sample < 129:
            audit = _audit_neutral_overlap(Path(right).resolve(), xml_root, local_meshes, world_poses=r)
            collision_free = audit['passed']
            collision_results.append({'sample': sample, 'passed': collision_free, 'overlaps': audit['overlaps']})
        for leaf in leaves:
            if collision_free:
                collision_free_endpoints.setdefault(leaf, []).append(r[leaf][:3, :3] @ offsets[leaf] + r[leaf][:3, 3])
            endpoints.setdefault(leaf, []).append(r[leaf][:3, :3] @ offsets[leaf] + r[leaf][:3, 3])
    report = {'sample_count': 1001, 'joint_count': len(moving), 'mirror_max_transform_error': peak,
              'mirror_passed': peak < 1e-9,
              'collision_sample_count': len(collision_results),
              'collision_free_sample_count': sum(x['passed'] for x in collision_results),
              'collision_samples': collision_results,
              'sampled_collision_free_fingertip_workspace_m': {name: {'min': np.min(points, axis=0).tolist(), 'max': np.max(points, axis=0).tolist()} for name, points in collision_free_endpoints.items()},
              'scope': 'Distal collision-surface cap points; uniform joint-space sampling including zero. All-sample bounds are not collision-filtered; separate bounds use 129 source-solid collision audits. Neither bounds certify continuous or dynamic reachability.',
              'fingertip_workspace_m': {name: {'min': np.min(points, axis=0).tolist(), 'max': np.max(points, axis=0).tolist()} for name, points in endpoints.items()}}
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'collision_samples', 'fingertip_workspace_m', 'sampled_collision_free_fingertip_workspace_m'}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('right'); parser.add_argument('left'); parser.add_argument('output')
    args = parser.parse_args(); main(args.right, args.left, args.output)
