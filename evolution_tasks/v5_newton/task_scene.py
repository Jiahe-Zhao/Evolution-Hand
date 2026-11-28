"""Create the four task scenes in Isaac Lab 3 without importing legacy Kit code."""
import re
import xml.etree.ElementTree as ET
import isaaclab.sim as sim
from isaaclab.assets import ArticulationCfg,AssetBaseCfg,RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from controller_config import make_actuators
from tactile_adapter import configure


def xyzw(q):return tuple(q[1:]+q[:1])


def create_scene_config(config,asset,urdf,num_envs):
    task=config['task'];source=config['source_contract'];native=source.get('config',{})
    xml=ET.parse(urdf).getroot()
    q={j.get('name'):max(float(j.find('limit').get('lower')),min(0.,float(j.find('limit').get('upper')))) for j in xml.findall('joint') if j.get('type') in ('revolute','prismatic')}
    if task=='grasp':
        q.update(config['scene']['initial_joint_positions']);pos=config['scene']['root_position'];rot=config['scene']['root_quaternion_xyzw']
    else:
        initial=native['robot_cfg']['init_state'];pos=initial['pos'];rot=xyzw(initial['rot'])
        for pattern,value in initial['joint_pos'].items():
            for name in q:
                if re.fullmatch(pattern,name):q[name]=value
    scene=InteractiveSceneCfg(num_envs=num_envs,env_spacing=1.)
    scene.hand=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Hand',
        spawn=sim.UsdFileCfg(usd_path=str(asset),articulation_props=NewtonArticulationCfg(self_collision_enabled=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pos),rot=tuple(rot),joint_pos=q),actuators=make_actuators(config,urdf))
    scene.light=AssetBaseCfg(prim_path='/World/Light',spawn=sim.DomeLightCfg(intensity=2000.))
    scene.ground=AssetBaseCfg(prim_path='/World/Ground',spawn=sim.GroundPlaneCfg())
    objects={}
    def add(name,size=None,pos=(0,0,0),rot=(0,0,0,1),mass=.02,static=False,radius=None,height=None,collision=True,friction=.5):
        kwargs=dict(mass_props=sim.MassCfg(mass=mass),rigid_props=sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=static),
            collision_props=sim.UsdPhysicsCollisionCfg(collision_enabled=collision),activate_contact_sensors=True,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.45,.30,.12)),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=friction,dynamic_friction=friction,restitution=0.))
        spawn=sim.CylinderCfg(radius=radius,height=height,axis='Z',**kwargs) if radius is not None else sim.CuboidCfg(size=tuple(size),**kwargs)
        cfg=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/'+name,spawn=spawn,init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos),rot=tuple(rot)))
        setattr(scene,name,cfg);objects[name]=cfg
    if task=='grasp':
        for row in config['scene']['objects']:
            add(row['name'],row['size'],row['position'],mass=row.get('mass',1.),static=row.get('static',False))
        target_names=['Object'];pair=('Object',['Support','Plate'])
    else:
        keys={'branch':['branch_cfg'],'forage':['food_cfg','leaf_one_cfg','leaf_two_cfg'],'strike':['Cone_cfg','strike_object_cfg']}[task]
        names={'branch_cfg':'branch','food_cfg':'food','leaf_one_cfg':'leaf_one','leaf_two_cfg':'leaf_two','Cone_cfg':'Cone','strike_object_cfg':'strike_object'}
        for key in keys:
            row=native[key];spawn=row['spawn'];initial=row['init_state'];props=spawn.get('mass_props') or {};mass=props.get('mass')
            if mass is None:
                import math
                volume=math.prod(spawn['size']) if 'size' in spawn else math.pi*spawn['radius']**2*spawn['height']
                mass=(props.get('density') or 1000.)*volume
            add(names[key],spawn.get('size'),initial['pos'],xyzw(initial['rot']),mass,
                static=bool((spawn.get('rigid_props') or {}).get('kinematic_enabled')),
                radius=spawn.get('radius'),height=spawn.get('height'),collision=(spawn.get('collision_props') or {}).get('collision_enabled') is not False,
                friction=.8 if task=='branch' else .5)
        # Food is a non-colliding target marker; leaves rest on the 0.10 m table.
        if task=='forage':add('table',(.7,.7,.10),(0,0,.05),mass=1.,static=True,friction=.5)
        if task=='strike':add('tool_support',(.012,.012,.020),(0,0,-.5),mass=1.,static=True,friction=1.)
        target_names={'branch':['branch'],'forage':['leaf_one','leaf_two','food'],'strike':['Cone']}[task]
        pair=('Cone',['strike_object']) if task=='strike' else None
    rows=configure(scene,asset,['{ENV_REGEX_NS}/'+name for name in target_names])
    if pair:
        scene.pair_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/'+pair[0],
            filter_prim_paths_expr=['{ENV_REGEX_NS}/'+name for name in pair[1]],track_friction_forces=True,history_length=1,update_period=0.)
    return scene,rows,objects,target_names
