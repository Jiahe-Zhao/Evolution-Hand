"""Size and place a 20-channel forearm tendon actuator packaging concept.

This artifact is external packaging, not additional simulated hand inertia.
Cable waypoints are routing proposals; moving cable clearance is unverified.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from audit_mechanical_workspace import graph, fk


def design(urdf, output):
    urdf, output = Path(urdf).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    nodes, root = graph(urdf)
    moving = [j for j in nodes if j.get('type') != 'fixed']
    poses = fk(nodes, root, {j.get('name'): 0. for j in moving})
    scene = trimesh.Scene()
    for link in ET.parse(urdf).getroot().findall('link'):
        for index, visual in enumerate(link.findall('visual')):
            spec = visual.find('geometry/mesh')
            mesh = trimesh.load(urdf.parent / spec.get('filename'), force='mesh')
            mesh.apply_scale(np.fromstring(spec.get('scale', '1 1 1'), sep=' '))
            mesh.visual.face_colors = [155, 180, 190, 255]
            # Compiled mechanical visuals have identity origins.
            origin = visual.find('origin')
            assert origin is None or all(float(x) == 0 for x in (origin.get('xyz', '0 0 0') + ' ' + origin.get('rpy', '0 0 0')).split())
            scene.add_geometry(mesh, node_name=f'{link.get("name")}_{index}', transform=poses[link.get('name')])
    channels = []
    for index, joint in enumerate(moving):
        row, column = divmod(index, 4)
        center = np.array([-0.16 + 0.025 * column, -0.04 + 0.02 * row, -0.025])
        envelope = trimesh.creation.cylinder(radius=0.008, height=0.020, sections=20)
        envelope.apply_translation(center)
        envelope.visual.face_colors = [45, 95, 140, 255]
        scene.add_geometry(envelope, node_name=f'actuator_envelope_{index}')
        guide = np.array([-0.045, center[1], -0.010])
        target = poses[joint.find('child').get('link')][:3, 3]
        paths = []
        for sign in (-1, 1):
            offset = np.array([0., sign * 0.002, 0.])
            points = np.array([center + [0, 0, 0.012] + offset, guide + offset, target + offset])
            route = trimesh.load_path(points)
            scene.add_geometry(route, node_name=f'tendon_{index}_{sign}')
            paths.append(points.tolist())
        limit = joint.find('limit')
        pulley_radius = 0.003
        channels.append({'joint': joint.get('name'), 'actuator_center_root_m': center.tolist(),
                         'actuator_envelope_radius_m': 0.008, 'actuator_envelope_length_m': 0.020,
                         'paired_tendon_waypoints_root_m': paths,
                         'joint_pulley_radius_m': pulley_radius,
                         'single_tendon_travel_m': pulley_radius * (float(limit.get('upper')) - float(limit.get('lower'))),
                         'differential_tension_limit_n': float(limit.get('effort')) / pulley_radius})
    centers = np.array([c['actuator_center_root_m'] for c in channels])
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    clearance = float(distances.min() - 0.016)
    assert clearance > 0
    base = trimesh.creation.box(extents=[0.106, 0.100, 0.004])
    base.apply_translation([-0.1225, 0, -0.037])
    base.visual.face_colors = [55, 65, 75, 255]
    scene.add_geometry(base, node_name='forearm_actuator_rack')
    scene.export(output / 'transmission_layout.glb')
    report = {'version': 1, 'source_urdf': str(urdf), 'joint_count': len(channels),
              'actuator_envelope_min_clearance_m': clearance, 'channels': channels,
              'status': 'packaging_concept',
              'limitations': ['Motor and gearbox models unselected; envelopes are design constraints.',
                              'Cable travel assumes ideal constant-radius joint pulleys.',
                              'Cable guides, pretension, friction and moving cable collision need hardware design.',
                              'External rack mass is not included in the fixed-base hand simulation.']}
    (output / 'transmission_layout.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'channels': len(channels), 'min_actuator_clearance_m': clearance}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('urdf'); parser.add_argument('output')
    args = parser.parse_args(); design(args.urdf, args.output)
