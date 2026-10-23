"""Import a verified standard BC run as generation zero, with an explicit contract."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from inheritance import policy_contract, check_seed_quality


def seed(run, source, output):
    run,source,output=Path(run),Path(source),Path(output)
    report=json.loads((run/'report.json').read_text())
    execution=json.loads((run/'execution.json').read_text())['args']
    if report['status']!='completed' or report['mode']!='bc':raise ValueError('Need a completed BC run')
    if report['source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('BC morphology differs')
    for key in ['history','target_skip','finger_skip','retrieval_bc','residual_bc','no_policy_time','no_target_input']:
        if execution.get(key):raise ValueError('Unsupported seed policy: '+key)
    if not report.get('separate_actor_critic'):raise ValueError('Need a separate actor and critic')
    check_seed_quality(0,report['before']['successes'],report['before']['num_envs'])
    joints=[j for j in ET.parse(source).getroot().findall('joint') if j.get('type')=='revolute']
    contract=policy_contract([j.get('name') for j in joints],[[float(j.find('limit').get(k)) for k in ['lower','upper']] for j in joints])
    output.mkdir(parents=True,exist_ok=False)
    checkpoint=output/'policy.pth';shutil.copyfile(run/'bc_initial.pth',checkpoint)
    (output/'policy_contract.json').write_text(json.dumps(contract,indent=2))
    parent=dict(id='baseline_g0',generation=0,code=[0]*8,checkpoint=str(checkpoint.resolve()),
                source_run=str(run.resolve()),bc_imported=True,bc_repeated=False,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (output/'parent.json').write_text(json.dumps([parent],indent=2));return parent


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(seed(a.run,a.source,a.output),indent=2))
