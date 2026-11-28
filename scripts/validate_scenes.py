#!/usr/bin/env python3
"""Execute each registered scene in a separate process and retain failure logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group',choices=['rigid','ax50','all','demo'],default='all');p.add_argument('--timeout',type=int,default=300);a=p.parse_args()
    specs=json.loads((ROOT/'configs/newton_scenes.json').read_text());out=ROOT/'outputs/portable_validation'/a.group;out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for key,spec in specs.items():
        soft=spec['backend']=='ax50_soft'
        if a.group=='rigid' and soft or a.group=='ax50' and not soft or a.group=='demo' and soft:continue
        for side in ['left','right'] if soft else ['left']:jobs.append((key,side,soft))
    def run(job):
        key,side,soft=job;name=key+'_'+side;log=out/(name+'.log');result=out/(name+'.json');start=time.monotonic()
        command=[sys.executable,str(ROOT/'scripts/run_scene.py'),key,'--side',side,'--result',str(result)]
        if a.group=='demo':command+=['--mode','demo']
        env=os.environ.copy();env['OMP_NUM_THREADS']='1'
        if not soft:env['CUDA_VISIBLE_DEVICES']=''
        with log.open('w') as f:
            try:returncode=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:returncode=124
        row={'scene':key,'side':side,'passed':returncode==0,'returncode':returncode,'elapsed_s':round(time.monotonic()-start,2),'log':str(log.relative_to(ROOT))}
        print(json.dumps(row),flush=True);return row
    # GPU soft worlds run sequentially to avoid exhausting VRAM; rigid cases can overlap.
    results=[]
    with ThreadPoolExecutor(max_workers=1 if a.group=="demo" else 3) as pool:results.extend(pool.map(run,[j for j in jobs if not j[2]]))
    for job in [j for j in jobs if j[2]]:results.append(run(job))
    report={'group':a.group,'scope':'installation and execution, not task-success qualification','passed':all(x['passed'] for x in results),'cases':results}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
