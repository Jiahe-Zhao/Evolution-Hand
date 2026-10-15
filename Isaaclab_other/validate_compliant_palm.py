"""Physical smoke test for the compliant palm; never treat mesh creation as a pass."""
import argparse
import faulthandler
import importlib.util
import json
import os
from pathlib import Path
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
print('[PALM] importing AppLauncher', flush=True)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--urdf', required=True)
parser.add_argument('--palm_assets', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--full_motion', action='store_true', help='Command a 12-second coordinated sweep to physical upper joint limits')
parser.add_argument('--palm_coordination', action='store_true')
parser.add_argument('--contact_probe', action='store_true', help='Indent the central palm web with a physical kinematic sphere')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject, DeformableObjectCfg
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
from compliant_palm_runtime import spawn_compliant_palm


def main():
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / 'hand_cfg.py'
    source = str(Path(args.urdf).resolve())
    parse_urdf_and_generate_articulation_cfg(source, source, str(cfg_path))
    spec = importlib.util.spec_from_file_location('palm_hand_cfg', cfg_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/240, device=args.device))
    hand_cfg = module.CURRENT_HAND_CFG.replace(prim_path='/World/Hand')
    hand_cfg = hand_cfg.replace(init_state=hand_cfg.init_state.replace(pos=(0., 0., 0.), rot=(1., 0., 0., 0.)))
    hand_cfg.spawn = hand_cfg.spawn.replace(make_instanceable=False, force_usd_conversion=True, usd_dir=str(out/'uninstanced_usd'))
    hand = Articulation(hand_cfg)
    print('[PALM] rigid hand spawned', flush=True)
    binding = spawn_compliant_palm(sim.stage, args.palm_assets)
    palm = DeformableObject(DeformableObjectCfg(prim_path='/World/CompliantPalm', spawn=None))
    sim.stage.Export(str(out/'authored_scene.usda'))
    probe = None
    if args.contact_probe:
        from isaaclab.assets import RigidObject, RigidObjectCfg
        probe_data = np.load(Path(args.palm_assets)/'palm_tetrahedra.npz')['points']
        # Central web between middle and ring supports, clear of embedded beams.
        probe_node = (9*13+6)*3
        probe_surface = probe_data[probe_node]
        probe_normal = probe_data[probe_node]-probe_data[probe_node+2]
        probe_normal = probe_normal/np.linalg.norm(probe_normal)
        probe_radius = 0.003
        probe_start = probe_surface + probe_normal*(probe_radius+0.004)
        probe = RigidObject(RigidObjectCfg(prim_path='/World/ContactProbe',
            spawn=sim_utils.SphereCfg(radius=probe_radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0003, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(probe_start))))
    print('[PALM] resetting simulation', flush=True)
    sim.reset(); hand.update(0); palm.update(0)
    rest = np.load(Path(args.palm_assets)/'palm_tetrahedra.npz')
    tet = torch.as_tensor(rest['tetrahedra'],device=hand.device,dtype=torch.long)
    points = torch.as_tensor(rest['points'],device=hand.device,dtype=torch.float32)
    if palm.data.nodal_pos_w.shape[1] != len(points):
        raise ValueError('PhysX simulation vertex layout differs from supplied tetrahedra')
    actual_rest = palm.data.default_nodal_state_w[0, :, :3]
    nearest = torch.cdist(actual_rest, points).min(dim=1)
    print('[PALM_REST_VERTEX_ERROR]', float((actual_rest-points).abs().max()), 'nearest_error', float(nearest.values.max()), 'unique_nearest', int(nearest.indices.unique().numel()), flush=True)
    def edges(x):
        q=x[tet]
        return torch.stack([q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]],dim=-1)
    inverse = torch.linalg.inv(edges(points))
    limits=hand.root_physx_view.get_dof_limits().to(hand.device) if args.full_motion else hand.data.soft_joint_pos_limits
    q0=torch.zeros_like(hand.data.joint_pos).clamp(limits[...,0],limits[...,1])
    target=q0.clone()
    for d,angle in zip(range(2,6),[0.25,0.30,0.40,0.50]):
        target[:,hand.joint_names.index(f'link_0_0_to_link_{d}_0')]=angle
    if args.full_motion:
        target = limits[...,1].clone()
    from mechanical_collision_audit import MechanicalCollisionAudit
    collision_auditor = MechanicalCollisionAudit(hand, source)
    collision_samples = []
    peak_limit_error = 0.
    hand.write_joint_state_to_sim(q0,torch.zeros_like(q0))
    max_stretch, min_jacobian, max_tracking = 1., 1., 0.
    from isaaclab.utils.math import quat_apply
    anchors = json.loads((Path(args.palm_assets)/'compliant_palm.json').read_text())['attachment_proposal']
    max_anchor_drift = 0.
    recorded_nodes, recorded_joints = [], []
    samples=[]
    probe_samples=[]
    motion_steps = 2880 if args.full_motion else 1440
    if args.full_motion:
        import xml.etree.ElementTree as ET
        velocity_by_name = {j.get('name'):float(j.find('limit').get('velocity')) for j in ET.parse(source).getroot().findall('joint') if j.get('type') != 'fixed'}
        velocity = torch.tensor([velocity_by_name[n] for n in hand.joint_names],device=hand.device)
        # A cosine out-and-back sweep has peak speed pi*amplitude/duration.
        # Leave acceleration/tracking margin rather than exceeding URDF speed.
        duration = max(12.,float((np.pi*(target-q0).abs()/velocity).max())/0.6)
        motion_steps = int(np.ceil(duration/sim.get_physics_dt()))
    for step in range(motion_steps+1):
        phase=(1-np.cos(2*np.pi*step/motion_steps))/2
        command=q0 if args.contact_probe else q0+phase*(target-q0)
        if probe is not None:
            probe_position = probe_surface + probe_normal*(probe_radius+0.004-phase*0.005)
            pose = torch.tensor([[*probe_position, 1., 0., 0., 0.]], device=hand.device, dtype=torch.float32)
            probe.write_root_pose_to_sim(pose)
        if args.palm_coordination:
            from palm_coordination import coordinate_palm_joints
            command = coordinate_palm_joints(command,list(hand.joint_names),limits)
        hand.set_joint_position_target(command);hand.write_data_to_sim()
        sim.step(render=False);hand.update(sim.get_physics_dt());palm.update(sim.get_physics_dt())
        x=palm.data.nodal_pos_w[0]
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite deformable state')
        max_tracking=max(max_tracking,float((hand.data.joint_pos-command).abs().max()))
        joint_pos=hand.data.joint_pos
        peak_limit_error=max(peak_limit_error,float(torch.maximum(limits[...,0]-joint_pos,joint_pos-limits[...,1]).clamp_min(0).max()))
        if step%120==0:
            collision_samples.append(dict(step=step, **collision_auditor.sample()))
        if step%60==0:
            if probe is not None:
                center = torch.as_tensor(probe_position, device=hand.device, dtype=torch.float32)
                gap = float(torch.linalg.vector_norm(x-center, dim=-1).min())-probe_radius
                depression = float(torch.dot(points[probe_node]-x[probe_node], torch.as_tensor(probe_normal, device=hand.device, dtype=torch.float32)))
                probe_samples.append({'step':step,'minimum_node_gap_m':gap,'central_depression_m':depression,'commanded_indentation_m':phase*0.005-0.004})
            anchor_errors = {}
            for name, anchor in anchors.items():
                body_id = hand.body_names.index(name)
                ids = torch.as_tensor(anchor['node_indices'], device=hand.device)
                local = torch.as_tensor(anchor['local_points_m'], device=hand.device, dtype=torch.float32)
                expected = quat_apply(hand.data.body_quat_w[0, body_id].expand(len(ids), -1), local) + hand.data.body_pos_w[0, body_id]
                anchor_errors[name] = float(torch.linalg.vector_norm(x[ids]-expected, dim=-1).max())
            max_anchor_drift = max(max_anchor_drift, max(anchor_errors.values()))
            recorded_nodes.append(x.detach().cpu().numpy())
            recorded_joints.append(hand.data.joint_pos[0].detach().cpu().numpy())
            deformation=edges(x)@inverse
            stretch=float(torch.linalg.svdvals(deformation).max())
            determinant=float(torch.linalg.det(deformation).min())
            max_stretch=max(max_stretch,stretch);min_jacobian=min(min_jacobian,determinant)
            samples.append({'step':step,'max_principal_stretch':stretch,'min_volume_ratio':determinant,'anchor_drift_m':anchor_errors})
            print('[PALM_SAMPLE]',json.dumps(samples[-1]),flush=True)
    # This smoke test checks basic dynamic integrity only. Contact/attachment
    # force transfer and task performance are separate acceptance requirements.
    report={'completed':True,'finite_state':True,'inverted_elements_detected':min_jacobian<=0,
            'max_principal_stretch':max_stretch,'min_volume_ratio':min_jacobian,
            'max_anchor_drift_m':max_anchor_drift,'max_joint_tracking_error_rad':max_tracking,'binding':binding,'samples':samples,
            'scope':'Dynamic integrity smoke test; full contact and attachment acceptance pending'}
    np.savez_compressed(out/'measured_deformation.npz', nodes=np.asarray(recorded_nodes), joints=np.asarray(recorded_joints), joint_names=np.asarray(hand.joint_names), rest_points=rest['points'], tetrahedra=rest['tetrahedra'])
    report['integrity_passed'] = bool(min_jacobian > 0 and max_anchor_drift <= 0.0015 and max_tracking <= 0.10)
    if probe_samples:
        peak = max(probe_samples, key=lambda v: v['commanded_indentation_m'])
        baseline = float(np.mean([v['central_depression_m'] for v in probe_samples if 60 <= v['step'] <= 300]))
        recovery = float(np.mean([v['central_depression_m'] for v in probe_samples if v['step'] >= 1200]))
        report['contact_probe'] = {'samples':probe_samples, 'peak':peak,
            'settled_baseline_depression_m':baseline,
            'additional_contact_depression_m':peak['central_depression_m']-baseline,
            'recovery_error_m':abs(recovery-baseline),
            'passed':bool(peak['central_depression_m']-baseline > 0.0002 and abs(peak['minimum_node_gap_m']) < 0.0003 and abs(recovery-baseline) < 0.0002 and min(v['minimum_node_gap_m'] for v in probe_samples) > -0.0007),
            'scope':'Kinematic spherical indentation, nodal separation and central deflection; not a contact-force or task-success test'}
        report['integrity_passed'] = report['integrity_passed'] and report['contact_probe']['passed']
    report['motion_duration_seconds'] = motion_steps*sim.get_physics_dt()
    report['full_motion_requested'] = args.full_motion
    report['palm_coordination'] = args.palm_coordination
    report['max_joint_limit_violation_rad'] = peak_limit_error
    report['collision_samples'] = collision_samples
    severe = [v for sample in collision_samples for v in sample['overlaps'] if v.get('severe')]
    report['source_solid_severe_overlap_count'] = len(severe)
    report['integrity_passed'] = report['integrity_passed'] and not severe and peak_limit_error <= 0.02
    report['integrity_thresholds'] = {'anchor_drift_m': 0.0015, 'tracking_rad': 0.10, 'minimum_volume_ratio': 0.0}
    (out/'dynamic_integrity.json').write_text(json.dumps(report,indent=2))
    print('[PALM_RESULT]',json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
    return report['integrity_passed']


try:
    passed = main()
except Exception:
    import traceback;traceback.print_exc()
    os._exit(1)
else:
    os._exit(0 if passed else 2)
