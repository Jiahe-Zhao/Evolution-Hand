#!/usr/bin/env python3
import argparse, hashlib, json, math
from pathlib import Path
import xml.etree.ElementTree as ET

def inspect(path):
    p = Path(path); root = ET.parse(p).getroot()
    links = {x.get('name'): x for x in root.findall('link')}
    joints = root.findall('joint'); errors = []; warnings = []
    if None in links or len(links) != len(root.findall('link')):
        errors.append('missing_or_duplicate_link')
    parents = {}; actuated = []
    for j in joints:
        parent_node, child_node = j.find('parent'), j.find('child')
        name = j.get('name')
        parent = parent_node.get('link') if parent_node is not None else None
        child = child_node.get('link') if child_node is not None else None
        kind = j.get('type')
        if not name or not parent or not child or child in parents or parent not in links or child not in links:
            errors.append('invalid_joint:' + str(name)); continue
        parents[child] = (parent, kind)
        if kind != 'fixed': actuated.append(name)
    roots = set(links) - set(parents)
    if len(roots) != 1: errors.append('root_count:' + str(len(roots)))
    for name in links:
        cur, seen = name, set()
        while cur in parents:
            if cur in seen: errors.append('cycle:' + name); break
            seen.add(cur); cur = parents[cur][0]
        if cur not in roots: errors.append('disconnected:' + name)
    mass = 0.0; collision_links = 0
    for name, link in links.items():
        inertial = link.find('inertial')
        if inertial is None: warnings.append('missing_inertial:' + name)
        else:
            node = inertial.find('mass'); value = float(node.get('value')) if node is not None else 0.0
            mass += value
            if value <= 0 or not math.isfinite(value): errors.append('bad_mass:' + name)
        if link.find('collision') is not None: collision_links += 1
        else: warnings.append('no_collision:' + name)
    unbounded = []
    for j in joints:
        if j.get('type') == 'fixed': continue
        limit = j.find('limit')
        if limit is None or limit.get('lower') is None or limit.get('upper') is None:
            unbounded.append(j.get('name')); continue
        lo, hi = float(limit.get('lower')), float(limit.get('upper'))
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi): errors.append('bad_limit:' + j.get('name'))
    if unbounded: warnings.append('unbounded_actuated:' + str(len(unbounded)))
    tips = sorted(n for n in links if any(k in n.lower() for k in ('tip', 'link_2_3', 'link_3_3', 'link_4_3', 'link_5_3')))
    return {'urdf': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'links': len(links), 'joints': len(joints), 'actuated_joints': len(actuated), 'collision_links': collision_links, 'roots': sorted(roots), 'tip_candidates': tips, 'total_mass': mass, 'errors': errors, 'warnings': warnings, 'pass': not errors and bool(actuated) and collision_links > 0}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('urdf'); parser.add_argument('--json'); args = parser.parse_args()
    result = inspect(args.urdf); print(json.dumps(result, indent=2))
    if args.json: Path(args.json).write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0 if result['pass'] else 2)
