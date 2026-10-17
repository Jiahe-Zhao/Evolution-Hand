"""Collect native-scene mechanical demonstrations; fail closed before BC."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,8,9])
    parser.add_argument('--tasks',nargs='+',default=['grasp','branch','strike','forage'])
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent.parent
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    assets=Path(args.assets).resolve()
    manifest={'version':1,'tasks':{}}
    for task in args.tasks:
        entries=[]
        for seed in args.seeds:
            destination=output/task/f'seed{seed}';destination.mkdir(parents=True,exist_ok=True)
            if task=='grasp':
                command=[sys.executable,str(repo/'evolution_tasks/task_grasp/mechanical_scripted_grasp.py'),
                    '--mechanical_asset_dir',str(assets),'--cartesian_replay','--joint_policy_demo',
                    '--closure_fraction','0.82','--contact_radius','0.045','--preflight_close_steps','300']
            else:
                command=[sys.executable,str(repo/'evolution_tasks/task_suite/mechanical_scripted_demo.py'),task,
                    '--mechanical_urdf',str(assets/'right/urdf/current_agent.urdf')]
                if task=='branch':command+=['--branch_policy_demo']
                if task=='strike':command+=['--strike_policy_demo','--strike_post_grasp_dwell_steps','40','--strike_max_steps','1200']
            command+=['--output',str(destination/'demo.mp4'),'--metrics',str(destination/'metrics.json'),
                      '--seed',str(seed),'--training_scene','--preflight','--headless','--device','cuda:0']
            environment=dict(os.environ,EVOLUTION_BRANCH_BC_MODE='1',EVOLUTION_STRIKE_BC_MODE='1')
            with (destination/'process.log').open('w') as log:
                result=subprocess.run(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{task} seed{seed} failed; inspect {destination}')
            report=json.loads((destination/'metrics.json').read_text())
            if not report['success'] or report.get('joint_limit_violation_steps',0):
                raise ValueError(f'{task} seed{seed} failed quality gate')
            entry={'metrics':str(destination/'metrics.json'),'trace':str(destination/'metrics.trace.npz'),
                   'scene':str(destination/'mechanical_scene_contract.json'),'seed':seed}
            for name in ('metrics','trace','scene'):entry[name+'_sha256']=digest(entry[name])
            entries.append(entry)
            print(task,seed,'passed',flush=True)
        manifest['tasks'][task]=entries
        (output/'manifest.partial.json').write_text(json.dumps(manifest,indent=2))
    if set(manifest['tasks'])=={'grasp','branch','forage','strike'}:
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
