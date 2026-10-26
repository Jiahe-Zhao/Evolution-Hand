import json,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent
root=b/'zero_extension_20261026';root.mkdir(exist_ok=False)
candidates=json.loads((b/'candidates.json').read_text());rows=[]
for name,index,seed in [('baseline_s2027',0,2027),('thumb_lateral_plus_s2028',1,2028)]:
 c=candidates[index];out=root/name
 cmd=[sys.executable,'-u',str(b/'zero_extension_worker.py'),'--source',c['source'],'--plans',c['plans'],'--output',str(out),'--resume',str(b/'runs'/name/'final.pth'),'--seed',str(seed),'--eval-seed','1062601','--num-envs','256','--steps','2097152','--eval-every','1048576','--headless']
 row=dict(name=name,status='running',command=cmd,started=time.time());rows.append(row)
 with (root/(name+'.log')).open('w') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid;(root/'status.json').write_text(json.dumps(rows,indent=2));code=p.wait()
 row.update(status='completed' if code==0 and (out/'final_evaluation/completed.json').exists() else 'failed',exit_code=code,duration_s=time.time()-row['started'])
 if (out/'final_evaluation/summary.json').exists():row['final_evaluation']=json.loads((out/'final_evaluation/summary.json').read_text())
 (root/'status.json').write_text(json.dumps(rows,indent=2));print(name,row['status'],flush=True)
lines=['# 零分策略追加 PPO 结果','', '每个策略从 4,980,736 步追加 2,097,152 步，固定学习率 3e-5。随机评估三个动作种子，均值动作另列。','']
for row in rows:
 lines.append('## '+row['name']+'：'+row['status'])
 for r in row.get('final_evaluation',[]):lines.append(f"- {r['label']}: {r['successes']}/{r['num_envs']}; 最大指部接触力 {r['diagnostics']['max_digit_force_N']:.2f} N；跌落环境数 {r['unique_fallen']}")
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
paths=[root/'RESULTS.md',root/'status.json']
for row in rows:
 out=root/row['name']
 for filename in ['report.json','effective_config.json','progress.json','learning_curve.jsonl','failure.txt','final_evaluation/summary.json','final_evaluation/manifest.json','final_evaluation/completed.json']:
  if (out/filename).exists():paths.append(out/filename)
repo=b.parent.parent
subprocess.run(['git','-C',str(repo),'add','-f','--']+[str(p) for p in paths],check=True)
subprocess.run(['git','-C',str(repo),'commit','--only','-m','Record bounded PPO extension outcomes','--']+[str(p) for p in paths],check=True)
print('EXTENSION_COMPLETED',flush=True)
