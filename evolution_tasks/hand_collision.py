"""Topology-derived filtering and versioned, shared train/eval contact setup."""
import xml.etree.ElementTree as ET
from pxr import Usd, UsdPhysics
import isaaclab.sim as sim_utils
from .collision_topology import collision_exclusions


@sim_utils.clone
def spawn_hand_with_collision_filters(prim_path, cfg, translation=None, orientation=None):
    prim = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation)
    bodies = {p.GetName(): p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    for a, b in collision_exclusions(cfg.asset_path, bodies):
        UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
    if 'motion_contact_v3_' in ET.parse(cfg.asset_path).getroot().get('name',''):
        from .mechanical_contact_materials import configure
        configure(prim.GetStage(), str(prim.GetPath()))
    return prim
