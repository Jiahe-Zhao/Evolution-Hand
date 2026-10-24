"""Sequential baseline + seven descendants; each gets the same full RL budget."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evolution import command, save
from genes import NAMES

BASE=Path(__file__).resolve().parent
CALIBRATION=BASE.parent/'contact_control_calibration_20261021'


def prepare_plan(source, output):
    """Retain the common initial plan when valid; otherwise explicitly replan."""
    import numpy as np
    sys.path.insert(0,str(CALIBRATION))
    from plan_grasp import Model
    model=Model(source,face='ymixed')
    reference=CALIBRATION/'plans_wrap_cover_selected.json'
    rows=json.loads(reference.read_text())
    def valid(row):
        q=np.array([row['q'][n] for n in model.names])
        if np.any(q<model.lo) or np.any(q>model.hi):return False
        hits=model.collisions(q,np.array(row['center']),np.array(row['rotation']))
        row['intersections']=hits
        return not hits
    accepted=[r for r in rows if valid(r)]
    replanned=False
    if not accepted:
        replanned=True
        temporary=output.with_suffix('.solver.json')
        with output.with_suffix('.planner.log').open('w') as log:
            subprocess.run([sys.executable,str(CALIBRATION/'plan_wrap_cover.py'),str(source),
                            str(reference),str(temporary),'ymixed'],stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=1200)
        accepted=[r for r in json.loads(temporary.read_text()) if valid(r)]
    if not accepted:raise ValueError('Initialization needs further planning; not a zero task score')
    save(output,accepted)
    save(output.with_suffix('.audit.json'),dict(source=str(source),replanned=replanned,valid_plans=len(accepted)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);a=parser.parse_args()
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    births=json.loads((BASE/'example_births.json').read_text())
    parent=json.loads((BASE/'seeds/g0/parent.json').read_text())[0]
    # Baseline control is an unmutated child of the same BC seed. It gets the
    # same fresh critic/optimizer and RL budget as its seven mutated siblings.
    baseline=dict(id='baseline_control',code=[0]*8,generation=1,parent_id=parent['id'],parent_checkpoint=parent['checkpoint'])
    rows=[baseline]+births
    state=dict(status='running',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               budget_per_individual=256*16*512,total_training_transitions=8*256*16*512,
               full_episode_steps=1500,bc_updates_allowed=0,individuals=[])
    save(root/'status.json',state)
    for index,row in enumerate(rows):
        record=copy.deepcopy(row);record.update(index=index,status='preparing');state['individuals'].append(record)
        save(root/'status.json',state)
        try:
            if index==0:candidate=BASE/'checks/baseline/candidate.json'
            else:
                changed=[(i,x) for i,x in enumerate(row['code']) if x]
                if len(changed)!=1:raise ValueError('Pilot expects single-gene neighbors')
                i,x=changed[0];candidate=BASE/'matrix'/f'{NAMES[i]}_{x:+d}'/'candidate.json'
            data=json.loads(candidate.read_text());plan=root/f'{index:02d}_plans.json'
            prepare_plan(Path(data['artifacts']['right']),plan)
            job=dict(row,candidate=str(candidate),plans=str(plan),mode='ppo')
            run=root/f'{index:02d}_{row["id"]}'
            args=command(job,run)
            save(root/f'{index:02d}_job.json',job);save(root/f'{index:02d}_command.json',args)
            record.update(status='running',run=str(run),started=time.time());save(root/'status.json',state)
            with run.with_suffix('.log').open('w') as log:
                process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
                record['worker_pid']=process.pid;save(root/'status.json',state)
                code=process.wait(timeout=14400)
            if code:raise RuntimeError(f'Worker exit {code}; see {run}.log')
            report=json.loads((run/'report.json').read_text())
            if report['bc_updates']!=0:raise RuntimeError('Unexpected descendant BC')
            record.update(status='completed',before_successes=report['before']['successes'],
                          after_successes=report['after']['successes'],episodes=report['after']['num_envs'],
                          duration_s=time.time()-record['started'])
            save(run/'lineage.json',job)
        except subprocess.TimeoutExpired:
            # Do not start another worker while a timed-out one is still alive.
            record.update(status='needs_attention',error='Timeout; no further jobs launched')
            state['status']='needs_attention';save(root/'status.json',state);raise
        except Exception as error:
            record.update(status='failed',error=f'{type(error).__name__}: {error}')
        save(root/'status.json',state)
    state['status']='completed' if all(r['status']=='completed' for r in state['individuals']) else 'completed_with_failures'
    save(root/'status.json',state)


if __name__=='__main__':main()
