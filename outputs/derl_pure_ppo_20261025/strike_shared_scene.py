"""Reconstruct task objects from the actual mechanical-hand training contract.

This deliberately does not import the legacy Shadow Strike scene. Hand placement,
reset clearance and policy adaptation remain separate validation gates.
"""
import copy,json,hashlib
from pathlib import Path

def load_contract(path):
 path=Path(path);cfg=json.loads(path.read_text())
 required=['Cone_cfg','strike_object_cfg','success_distance','success_force_threshold','tool_finger_contact_force_threshold','min_long_finger_contacts','tool_grasp_hold_steps']
 missing=[k for k in required if k not in cfg]
 if missing:raise ValueError('Missing native task fields: '+str(missing))
 contract={k:copy.deepcopy(cfg[k]) for k in required}
 contract.update(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tool_grasp_loss_grace_steps=cfg.get('tool_grasp_loss_grace_steps',2),sim_dt=cfg['sim']['dt'],decimation=cfg['decimation'],episode_length_s=cfg['episode_length_s'],support=dict(size=[.012,.012,.020],contact_offset=.001,rest_offset=0.,static_friction=1.,dynamic_friction=1.,kinematic=True,placement='under actual reset tool bottom; no tool attachment'),target_force_filter='tool only, not net target force from hand/support')
 return contract

def rigid_object_from_contract(row,prim_path):
 import isaaclab.sim as sim
 from isaaclab.assets import RigidObjectCfg
 def clean(d):return {k:v for k,v in (d or {}).items() if k!='func'}
 spawn=row['spawn']
 if not spawn['func'].endswith(':spawn_cuboid'):raise ValueError('Expected native cuboid, not legacy cone')
 cfg=sim.CuboidCfg(size=tuple(spawn['size']),rigid_props=sim.RigidBodyPropertiesCfg(**clean(spawn['rigid_props'])),collision_props=sim.CollisionPropertiesCfg(**clean(spawn['collision_props'])),mass_props=sim.MassPropertiesCfg(**clean(spawn['mass_props'])),activate_contact_sensors=spawn.get('activate_contact_sensors',False))
 if spawn.get('physics_material'):cfg.physics_material=sim.RigidBodyMaterialCfg(**clean(spawn['physics_material']))
 if spawn.get('visual_material'):cfg.visual_material=sim.PreviewSurfaceCfg(**clean(spawn['visual_material']))
 return RigidObjectCfg(prim_path=prim_path,spawn=cfg,init_state=RigidObjectCfg.InitialStateCfg(**row['init_state']))

def native_hit(held,holding_now,tip,target,filtered_tool_target_force,contract):
 """Same geometric/contact predicate as the current native _get_rewards."""
 return held & holding_now & ((tip[:,:2]-target[:,:2]).norm(dim=-1)<contract['success_distance']) & (filtered_tool_target_force.norm(dim=-1)>=contract['success_force_threshold'])

SHADOW_TIPS=['robot0_thdistal','robot0_ffdistal','robot0_mfdistal','robot0_rfdistal','robot0_lfdistal']

def shadow_digit_groups(body_names):
 prefixes=['robot0_th','robot0_ff','robot0_mf','robot0_rf','robot0_lf']
 groups=[[i for i,n in enumerate(body_names) if n.startswith(prefix)] for prefix in prefixes]
 if any(not g for g in groups):raise ValueError('Missing Shadow digit bodies')
 return groups

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
 Path(a.output).write_text(json.dumps(load_contract(a.source),indent=2))
