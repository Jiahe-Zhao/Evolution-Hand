"""Create all four BC checkpoints before starting mechanical evolution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    base=Path(__file__).resolve().parent
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    assets=Path(args.assets).resolve()
    legacy=base/'evaluation_results/preflight_grasp_projected_20260922/morphology/0_0/right/urdf/current_agent.urdf'
    seeds={'version':1,'demonstration_manifest':str(Path(args.manifest).resolve()),'tasks':{}}
    for task in ('grasp','branch','forage','strike'):
        report=json.loads((base/f'mechanical_mapping_v4/ppo_integration/{task}4/mechanical_training.json').read_text())
        destination=output/task;destination.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(base/'evaluate_mechanical_task.py'),'--task',task,
            '--checkpoint',report['source_checkpoint'],'--curriculum_stage','stage2',
            '--legacy_source_urdf',str(legacy),'--mechanical_asset_dir',str(assets),
            '--lineage_json',str(assets.parent/'lineage.json'),'--individual_key','candidate',
            '--output_dir',str(destination),'--headless','--train_iterations','1',
            '--train_num_envs','4','--train_from_scratch','--bc_only','--bc_manifest',str(Path(args.manifest).resolve())]
        if task in ('grasp','forage'):command+=['--mechanical_joint_actions']
        with (destination/'process.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        training=json.loads((destination/'mechanical_training.json').read_text())
        if training['status']!='completed' or not training['bc_only'] or not training['all_model_tensors_finite']:
            raise ValueError('Incomplete or nonfinite BC seed')
        bc=training['behavior_cloning']
        if bc['final_loss']>=bc['first_loss']:
            raise ValueError('BC did not reduce action prediction loss')
        checkpoint=Path(training['checkpoint'])
        seeds['tasks'][task]={'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              'training_report':str(destination/'mechanical_training.json')}
        (output/'seeds.partial.json').write_text(json.dumps(seeds,indent=2))
        print(task,'BC checkpoint complete',bc['final_loss'],flush=True)
    (output/'seeds.json').write_text(json.dumps(seeds,indent=2))

if __name__=='__main__':main()
