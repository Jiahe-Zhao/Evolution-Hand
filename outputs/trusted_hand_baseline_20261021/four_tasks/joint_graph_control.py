"""Morphology-native joint channels; no per-finger or total-joint cap.

This contract is deliberately versioned separately from fixed 20-channel BC.
Environment adapters must construct action/observation spaces from this layout.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def vector(text, default):
    result = tuple(float(x) for x in (text or default).split())
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError('Expected finite xyz vector')
    return result


@dataclass(frozen=True)
class Joint:
    name: str
    parent_link: str
    child_link: str
    parent_joint: str | None
    depth: int
    kind: str
    lower: float
    upper: float
    effort: float
    velocity: float
    axis: tuple
    origin_xyz: tuple
    origin_rpy: tuple


@dataclass(frozen=True)
class JointGraph:
    joints: tuple
    urdf_sha256: str

    @classmethod
    def from_urdf(cls, path):
        content = Path(path).read_bytes()
        root = ET.fromstring(content)
        links = [node.attrib['name'] for node in root.findall('link')]
        nodes = root.findall('joint')
        names = [node.attrib['name'] for node in nodes]
        if len(set(links)) != len(links) or len(set(names)) != len(names):
            raise ValueError('Duplicate link or joint name')
        incoming = {}
        for node in nodes:
            parent = node.find('parent').attrib['link']
            child = node.find('child').attrib['link']
            if parent not in links or child not in links or child in incoming:
                raise ValueError('Invalid or multiply-parented link')
            incoming[child] = node
        def ancestry(node):
            chain = []
            seen = {node.attrib['name']}
            parent = node.find('parent').attrib['link']
            while parent in incoming:
                ancestor = incoming[parent]
                if ancestor.attrib['name'] in seen:
                    raise ValueError('Cyclic joint graph')
                seen.add(ancestor.attrib['name'])
                chain.append(ancestor)
                parent = ancestor.find('parent').attrib['link']
            return chain
        rows = []
        for node in nodes:
            ancestors = ancestry(node)  # Also validate fixed-only chains.
            kind = node.attrib['type']
            if kind == 'fixed':
                continue
            if node.find('mimic') is not None:
                raise ValueError('Mimic joint requires an explicit coupling adapter')
            if kind not in ('revolute', 'prismatic'):
                raise ValueError('Unbounded/floating joint needs a separate control convention')
            limit = node.find('limit')
            if limit is None:
                raise ValueError('Actuated joint has no physical limits')
            lo, hi, effort, velocity = (float(limit.attrib[k]) for k in ('lower','upper','effort','velocity'))
            if not all(math.isfinite(v) for v in (lo,hi,effort,velocity)) or hi <= lo or min(effort,velocity) <= 0:
                raise ValueError('Invalid physical joint limits')
            active_ancestors = [a for a in ancestors if a.attrib['type'] != 'fixed']
            axis = node.find('axis'); origin = node.find('origin')
            direction = vector(axis.get('xyz') if axis is not None else None, '1 0 0')
            norm = math.sqrt(sum(x*x for x in direction))
            if norm == 0:
                raise ValueError('Zero joint axis')
            rows.append(Joint(node.attrib['name'], node.find('parent').attrib['link'],
                              node.find('child').attrib['link'],
                              active_ancestors[0].attrib['name'] if active_ancestors else None,
                              len(active_ancestors), kind, lo, hi, effort, velocity,
                              tuple(x/norm for x in direction),
                              vector(origin.get('xyz') if origin is not None else None, '0 0 0'),
                              vector(origin.get('rpy') if origin is not None else None, '0 0 0')))
        if not rows:
            raise ValueError('No actuated joints')
        rows.sort(key=lambda j: (j.depth, j.name))
        return cls(tuple(rows), hashlib.sha256(content).hexdigest())

    def contract(self, wrist=False):
        actions = [f'joint:{j.name}.target' for j in self.joints]
        if wrist:
            actions += [f'wrist.delta.{axis}' for axis in 'xyz']
        return dict(version=2, controller='joint_graph_v2', actions=actions,
                    joint_names=[j.name for j in self.joints],
                    joints=[asdict(j) for j in self.joints],
                    urdf_sha256=self.urdf_sha256)

    def simulator_indices(self, actual_names):
        if len(set(actual_names)) != len(actual_names):
            raise ValueError('Duplicate simulator joint names')
        expected = {j.name for j in self.joints}
        if expected != set(actual_names):
            raise ValueError(f'Unmapped joints: URDF-only={expected-set(actual_names)}, simulator-only={set(actual_names)-expected}')
        indices = {name: index for index, name in enumerate(actual_names)}
        return [indices[j.name] for j in self.joints]

    def decode(self, normalized):
        if len(normalized) != len(self.joints):
            raise ValueError('Action dimension mismatch')
        if not all(math.isfinite(float(x)) for x in normalized):
            raise ValueError('Non-finite action')
        return [j.lower + (max(-1.,min(1.,float(a)))+1)*0.5*(j.upper-j.lower)
                for j,a in zip(self.joints,normalized)]

    def encode(self, positions):
        if len(positions) != len(self.joints):
            raise ValueError('Position dimension mismatch')
        if not all(math.isfinite(float(x)) for x in positions):
            raise ValueError('Non-finite position')
        return [max(-1.,min(1.,2*(float(q)-j.lower)/(j.upper-j.lower)-1))
                for j,q in zip(self.joints,positions)]

    def inheritance_map(self, parent):
        """Common channels transfer; added channels remain explicitly new."""
        old = {j.name: i for i,j in enumerate(parent.joints)}
        return dict(common=[(i,old[j.name]) for i,j in enumerate(self.joints) if j.name in old],
                    added=[i for i,j in enumerate(self.joints) if j.name not in old],
                    removed=[i for i,j in enumerate(parent.joints) if j.name not in {x.name for x in self.joints}])


class JointTargetController:
    """Batched targets in simulator order, with physical velocity limits.

Call once per physics step with that step's dt. Previous *commanded* targets
are state, not measured positions; this prevents target jumps under contact.
Wrist control is separate and is not gated by a prescribed finger subset.
"""
    def __init__(self, graph, simulator_names, device, dtype=None, external_joint_names=()):
        import torch
        self.graph = graph
        if external_joint_names:
            expected={j.name for j in graph.joints}
            if set(simulator_names)!=expected|set(external_joint_names) or expected&set(external_joint_names) or len(simulator_names)!=len(set(simulator_names)):
                raise ValueError('Unmapped external or morphology joints')
            mapped=[simulator_names.index(j.name) for j in graph.joints]
        else:mapped=graph.simulator_indices(simulator_names)
        self.simulator_joint_count=len(simulator_names)
        self.indices = torch.tensor(mapped, device=device)
        dtype = dtype or torch.float32
        self.lower = torch.tensor([j.lower for j in graph.joints], device=device, dtype=dtype)
        self.upper = torch.tensor([j.upper for j in graph.joints], device=device, dtype=dtype)
        self.velocity = torch.tensor([j.velocity for j in graph.joints], device=device, dtype=dtype)

    def targets(self, actions, previous_targets, dt):
        import torch
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Require positive finite physics dt')
        if actions.shape[:-1] != previous_targets.shape[:-1] or actions.shape[-1] != len(self.graph.joints) or previous_targets.shape[-1] != self.simulator_joint_count:
            raise ValueError('Require matching batch and joint dimensions')
        if not bool(torch.isfinite(actions).all()) or not bool(torch.isfinite(previous_targets).all()):
            raise ValueError('Non-finite controller input')
        previous = previous_targets.index_select(-1, self.indices)
        if bool(((previous < self.lower-1e-6) | (previous > self.upper+1e-6)).any()):
            raise ValueError('Previous command is outside physical limits; reset explicitly')
        desired = self.lower + (actions.clamp(-1,1)+1)*0.5*(self.upper-self.lower)
        bounded = previous + (desired-previous).clamp(-self.velocity*dt,self.velocity*dt)
        output = previous_targets.clone()
        output[..., self.indices] = bounded.clamp(self.lower,self.upper)
        return output

    def joint_observations(self, positions, velocities):
        q = positions.index_select(-1,self.indices)
        dq = velocities.index_select(-1,self.indices)
        # Do not clip measured overshoot: observations must retain violations.
        return 2*(q-self.lower)/(self.upper-self.lower)-1, dq/self.velocity
