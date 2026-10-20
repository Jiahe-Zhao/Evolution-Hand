"""Shared mechanical-task scene corrections, applied before environment creation."""
import hashlib
import json
import os
from pathlib import Path


def apply_mechanical_scene(cfg, task, output_dir):
    changes = {}
    urdf = Path(cfg.robot_cfg.spawn.asset_path).resolve()
    if task == 'grasp':
        old_radius=cfg.grasp_object_cfg.spawn.radius
        cfg.grasp_object_cfg.spawn.radius=0.045
        cfg.grasp_object_radius=0.045
        old_offset=tuple(cfg.proximal_support_offset_local)
        cfg.proximal_support_offset_local=(old_offset[0],old_offset[1],-0.020)
        changes['object_radius']={'before':old_radius,'after':0.045}
        changes['support_offset']={'before':old_offset,'after':cfg.proximal_support_offset_local}
    if task == 'grasp' and urdf.parent.parent.name == 'left':
        # The existing task constants describe the right-hand root frame.
        # Mirror local scene offsets/regions along with the left-hand asset.
        for name in ('proximal_support_offset_local','visual_palm_region_center','proximal_finger_region_center'):
            old = tuple(getattr(cfg,name)); new = (-old[0],*old[1:])
            setattr(cfg,name,new);changes[name]={'before':old,'after':new}
    if task == 'forage':
        changes['leaf_planar_support']='passive_D6_xy_free_z_rotation_locked'
        changes['leaf_linear_damping']=5.0
        cfg.leaf_one_cfg.spawn.rigid_props.linear_damping=5.0
        cfg.leaf_two_cfg.spawn.rigid_props.linear_damping=5.0
        cfg.leaf_one_cfg.spawn.rigid_props.disable_gravity=False
        cfg.leaf_two_cfg.spawn.rigid_props.disable_gravity=False
        changes['wrist_translation_limits']={'before':tuple(cfg.wrist_translation_limits),'after':(0.20,0.20,0.08)}
        cfg.wrist_translation_limits=(0.20,0.20,0.08)
        changes['food_support']={'before_size':tuple(cfg.food_cfg.spawn.size),'after_size':(0.050,0.045,0.040),'top_height_m':0.080}
        cfg.food_cfg.spawn.size=(0.050,0.045,0.040)
        position=tuple(cfg.food_cfg.init_state.pos)
        cfg.food_cfg.init_state=cfg.food_cfg.init_state.replace(pos=(position[0],position[1],0.060))
        old=dict(cfg.robot_cfg.init_state.joint_pos or {})
        opened={name:(value if name.startswith('link_1_') or name=='link_0_0_to_link_1_0' else 0.0) for name,value in old.items()}
        import xml.etree.ElementTree as ET
        for joint in ET.parse(urdf).getroot().findall('joint'):
            name=joint.get('name')
            if joint.get('type')!='fixed' and 'spread' not in name and any(f'link_{finger}_' in name for finger in range(2,6)):
                opened[name]=0.10
        cfg.robot_cfg=cfg.robot_cfg.replace(init_state=cfg.robot_cfg.init_state.replace(joint_pos=opened))
        props=cfg.robot_cfg.spawn.articulation_props
        changes['solver_iterations']={'before':[props.solver_position_iteration_count,props.solver_velocity_iteration_count],'after':[64,8]}
        props.solver_position_iteration_count=64
        props.solver_velocity_iteration_count=8
        changes['initial_joint_positions']={'before':old,'after':opened}
        changes['episode_length_s']={'before':cfg.episode_length_s,'after':40.0}
        cfg.episode_length_s=40.0
    if task == 'strike':
        preset=os.environ.get('EVOLUTION_STRIKE_SCENE_PRESET','clearance_v3')
        if preset not in ('clearance_v3','open_grasp_v4'):raise ValueError('Unknown Strike scene preset: '+preset)
        if preset=='open_grasp_v4':
            changes['scene_preset']=preset
            width=float(os.environ.get('EVOLUTION_STRIKE_V4_TOOL_WIDTH_M','0.025'))
            pregrasp=float(os.environ.get('EVOLUTION_STRIKE_V4_PREGRASP','-0.20'))
            if not 0.012<=width<=0.03 or not -0.5<=pregrasp<=0.45:raise ValueError('Strike v4 scene parameters outside supported range')
            old_size=tuple(cfg.Cone_cfg.spawn.size)
            cfg.Cone_cfg.spawn.size=(width,width,old_size[2])
            changes['tool_size_m']={'before':old_size,'after':cfg.Cone_cfg.spawn.size}
        reset={'EVOLUTION_STRIKE_RESET_THUMB_SPREAD':'-0.80',
               'EVOLUTION_STRIKE_PREGRASP_ACTION':(str(pregrasp) if preset=='open_grasp_v4' else '0.45'),
               'EVOLUTION_STRIKE_RESET_OFFSET_WORLD':'0,0,0'}
        os.environ.update(reset)
        os.environ.pop('EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT',None)
        changes['reset_environment']=reset
        changes['episode_length_s']={'before':cfg.episode_length_s,'after':20.0}
        cfg.episode_length_s = 20.0
        old = cfg.wrist_max_step_m
        cfg.wrist_max_step_m = 0.00005
        changes['wrist_max_step_m']={'before':old,'after':cfg.wrist_max_step_m}
        cfg.tool_grasp_loss_grace_steps = 2
        cfg.joint_target_speed_limit_rad_s = 0.45
        cfg.tool_reset_clearance_enabled = True
        changes['tool_grasp_loss_grace_steps'] = 2
        changes['joint_target_speed_limit_rad_s'] = 0.45
        changes['tool_reset_clearance'] = {'enabled': True, 'margin_m': 0.001,
            'search_xy_half_extent_m': 0.030, 'grid_m': 0.001,
            'geometry': 'imported_collision_mesh_convex_hulls', 'failure': 'reject_reset'}
    effective={k:(v['after'] if isinstance(v,dict) and 'after' in v else v) for k,v in changes.items()}
    if 'food_support' in effective:
        effective['food_support']={k:v for k,v in effective['food_support'].items() if not k.startswith('before')}
    contract={'version':(4 if task=='strike' and preset=='open_grasp_v4' else 3),'effective_settings':effective,'curriculum_stage':cfg.curriculum_stage,'task':task,'changes':changes,
              'mechanical_urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest(),
              'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'success_thresholds_unchanged':True}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'mechanical_scene_contract.json').write_text(json.dumps(contract,indent=2))
    return contract
