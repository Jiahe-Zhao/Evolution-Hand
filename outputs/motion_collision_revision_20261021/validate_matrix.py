"""Exercise actual evolution operators and mirrored kinematics, without RL."""
import copy, json, sys, random, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/zjh/Evolution_PC/Isaaclab_other')
sys.path.insert(0,str(BASE/'patch'))
import tools
from code_to_urdf import generate_urdf_from_dict
from compile_mechanical_hand import compile_hand
from mirror_mechanical_urdf import mirror
from collision_gate import audit_generated_urdf, validate_morphology
from mechanical_contact_model import pose


def fk(path,q):
    root=ET.parse(path).getroot();todo=root.findall('joint')
    children={j.find('child').get('link') for j in todo}
    ts={next(l.get('name') for l in root.findall('link') if l.get('name') not in children):np.eye(4)}
    while todo:
        ready=[j for j in todo if j.find('parent').get('link') in ts]
        if not ready:raise ValueError('disconnected tree')
        for j in ready:
            t=pose(j.find('origin'));r=np.eye(4)
            if j.get('type')=='revolute':r[:3,:3]=Rotation.from_rotvec(np.fromstring(j.find('axis').get('xyz'),sep=' ')*q.get(j.get('name'),0.)).as_matrix()
            ts[j.find('child').get('link')]=ts[j.find('parent').get('link')]@t@r;todo.remove(j)
    return ts


def validate_motion(right,left):
    r=ET.parse(right).getroot();rng=np.random.default_rng(8);M=np.diag([-1.,1.,1.]);maximum=0.;axes=[]
    q={j.get('name'):float(rng.uniform(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))) for j in r.findall('joint') if j.get('type')=='revolute'}
    a,b=fk(right,q),fk(left,q)
    for name,t in a.items():maximum=max(maximum,float(np.max(abs(b[name][:3,3]-M@t[:3,3]))),float(np.max(abs(b[name][:3,:3]-M@t[:3,:3]@M))))
    assert maximum<1e-8,maximum
    for d in range(2,6):
        name=f'link_{d}_mcp_spread_joint';j=r.find(f"joint[@name='{name}']")
        if j is None:continue
        childs=[l.get('name') for l in r.findall('link') if l.get('name').startswith(f'link_{d}_') and l.get('name').split('_')[-1].isdigit()]
        tip=max(childs,key=lambda s:int(s.split('_')[-1]));local=np.array([0.,0.,.005,1.]);p=(a[tip]@local)[:3]
        h=q.copy();h[name]+=1e-5;derivative=((fk(right,h)[tip]@local)[:3]-p)/1e-5
        # Independent MCP axes must move a distal point, not rotate redundantly.
        axes.append({'joint':name,'tip_speed_per_rad':float(np.linalg.norm(derivative))})
        assert np.linalg.norm(derivative)>1e-4
    return {'mirror_max_error':maximum,'spread_motion':axes,'actuated_count':len(q)}


def candidates(hand):
    yield 'baseline',hand,'baseline'
    for op in ['change_link_length','change_link_radius','change_thumb_length','change_palm_curvature','change_joint_origin_translation','change_joint_origin_rpy']:
        for sign in [-1,1]:
            h=copy.deepcopy(hand);random.seed(123);np.random.seed(123)
            if op=='change_thumb_length':h=getattr(tools,op)(h,sign*.15)
            elif op=='change_palm_curvature':h=getattr(tools,op)(h,sign*.05)
            else:h=getattr(tools,op)(h,'link_2_1' if op!='change_joint_origin_rpy' else 'link_2_3',sign*(.15 if op in ['change_link_length','change_link_radius'] else .03))
            if not isinstance(h,str):tools.synchronize_kinematic_connections(h)
            yield op+('_minus' if sign<0 else '_plus'),h,op
    for op in ['add_link','remove_link']:
        for target in ([max((l['name_code'] for l in hand['links'] if l['name_code'].startswith(f'link_{d}_')),key=lambda n:int(n.split('_')[-1])) for d in [1,2]] if op=='add_link' else ['link_2_1','link_3_1']):
            h=getattr(tools,op)(copy.deepcopy(hand),target)
            if not isinstance(h,str):tools.synchronize_kinematic_connections(h)
            yield op+'_'+target,h,op


def main():
    cmd=json.loads(Path('/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021/four_task_results/strike_7_v2/command.json').read_text())['args']
    source=Path(cmd[cmd.index('--lineage_json')+1]);hand=json.loads(source.read_text())['lineage']['candidate']['urdf_info']
    (BASE/'source_genotype.json').write_text(json.dumps(hand,indent=2))
    records=[]
    protected=tools.change_joint_origin_rpy(copy.deepcopy(hand),'link_2_0',.03)
    (BASE/'mutation_contract.json').write_text(json.dumps({'protected_flexion_origin_rpy_unchanged':protected==hand,'rotation_matrix_test_target':'link_2_3','flexion_positive_direction':'unchanged','spread_axis':'independent local Y','limits':'original generated limits retained','old_checkpoints':'require retraining or explicit revalidation after axis change'},indent=2))
    for label,h,op in candidates(hand):
        row={'name':label,'operation':op};records.append(row)
        try:
            if isinstance(h,str):raise ValueError(h)
            if op!='baseline' and h==hand:raise ValueError('Mutation produced no genotype change')
            valid,top=validate_morphology(h)
            if not valid:raise ValueError('source topology invalid: '+str(top.get('reasons')))
            out=BASE/'matrix'/label;out.mkdir(parents=True,exist_ok=True)
            (out/'genotype.json').write_text(json.dumps(h,indent=2))
            bone=out/'bone/urdf/current_agent.urdf';generate_urdf_from_dict(h,str(out/'bone/meshes'),str(bone))
            right=out/'right/urdf/current_agent.urdf';left=out/'left/urdf/current_agent.urdf'
            mapping=compile_hand(bone,right);mirror(right,left)
            passed,audit=audit_generated_urdf(right)
            assert passed,audit.get('reasons')
            passed,audit=audit_generated_urdf(left)
            assert passed,audit.get('reasons')
            motion=validate_motion(right,left)
            assert all(abs(x['flexion_dot'])<1e-6 for x in mapping['spread_axes'])
            row.update(status='passed',motion=motion,mass_kg=mapping['mass_kg'],right=str(right),left=str(left),
                       collision_count=len(ET.parse(right).getroot().findall('link/collision')),
                       source_sha256=mapping['source_sha256'],urdf_sha256=mapping['urdf_sha256'])
        except Exception as e:row.update(status='failed',reason=f'{type(e).__name__}: {e}')
        (BASE/'matrix.json').write_text(json.dumps(records,indent=2));print(label,row['status'],row.get('reason',''),flush=True)

if __name__=='__main__':main()
