"""Map legacy actuator configuration explicitly to Isaac Lab 3 fields."""
import re
import xml.etree.ElementTree as ET

def actuator_parameters(config,urdf):
    joints=[j for j in ET.parse(urdf).getroot().findall('joint') if j.get('type') in ('revolute','prismatic')]
    names=[j.get('name') for j in joints]
    if config.get("paired_v5") or config['task']=='grasp' or config.get('actuator_profile'):
        fingers=[j for j in joints if not j.get('name').startswith(('diagnostic_wrist_','wrist_'))]
        control=dict(kp=3.,kd=.3,armature=.001) if config.get("paired_v5") else config['control'] if config['task']=='grasp' else dict(kp=config['actuator_profile']['hand']['stiffness'],kd=config['actuator_profile']['hand']['damping'],armature=config['actuator_profile']['hand']['armature'])
        result={'hand':dict(joint_names_expr=[re.escape(j.get('name')) for j in fingers],
             stiffness=control['kp'],damping=control['kd'],armature=control['armature'],
             joint_effort_limit={re.escape(j.get('name')):float(j.find('limit').get('effort')) for j in fingers},
             joint_velocity_limit={re.escape(j.get('name')):float(j.find('limit').get('velocity')) for j in fingers}),
             'wrist':dict(joint_names_expr=['diagnostic_wrist_.*'],stiffness=3000.,damping=100.,armature=0.,joint_effort_limit=100.,joint_velocity_limit=.02)}
    else:
        result={}
        for key,value in config['control']['actuators'].items():
            result[key]={k:value[k] for k in ['joint_names_expr','stiffness','damping','armature'] if value.get(k) is not None}
            for old,new in [('effort_limit_sim','joint_effort_limit'),('velocity_limit_sim','joint_velocity_limit')]:
                if value.get(old) is None:raise ValueError(f'Missing explicit {old}: {key}')
                result[key][new]=value[old]
    if any(n in ('wrist_roll','wrist_pitch','wrist_yaw') for n in names):
        result['wrist_rotation']=dict(joint_names_expr=['wrist_roll','wrist_pitch','wrist_yaw'],stiffness=2.,damping=.1,armature=.0001,joint_effort_limit=2.,joint_velocity_limit=1.)
    profile=config.get('actuator_profile')
    if profile:
        for group in ['hand','wrist']:
            fields=profile[group]
            if fields.get('dynamic_friction',0)!=0:raise ValueError('Newton does not support separate kinetic joint friction')
            for key in ['stiffness','damping','armature','friction','dynamic_friction','viscous_friction']:
                value=float(fields[key])
                if not __import__('math').isfinite(value) or value<0:raise ValueError((group,key,value))
                result[group][key]=value
    assigned={n:[] for n in names}
    for key,value in result.items():
        for name in names:
            if any(re.fullmatch(pattern,name) for pattern in value['joint_names_expr']):assigned[name].append(key)
    if any(len(v)!=1 for v in assigned.values()):raise ValueError(f'Unassigned or duplicate actuator joints: {assigned}')
    return result

def make_actuators(config,urdf):
    from isaaclab.actuators import ImplicitActuatorCfg
    return {name:ImplicitActuatorCfg(**values) for name,values in actuator_parameters(config,urdf).items()}
