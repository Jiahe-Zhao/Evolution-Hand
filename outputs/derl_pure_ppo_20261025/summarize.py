"""Report real outcomes only, including incomplete/failed jobs."""
import json,statistics
from pathlib import Path
b=Path(__file__).resolve().parent;root=b/'runs';state_file=root/'status.json'
if not state_file.exists():raise SystemExit('No full runs started')
state=json.loads(state_file.read_text());rows=[];evidence=b/'evidence';evidence.mkdir(exist_ok=True)
for job in state:
 folder=root/job['name'];r=dict(name=job['name'],status=job['status'])
 for name in ('report.json','progress.json'):
  if (folder/name).exists():
   data=json.loads((folder/name).read_text())
   if name=='report.json':
    r.update(successes=data['after']['successes'],episodes=data['after']['num_envs'],before=data['before']['successes'],best_successes=data['best_successes'],frames=data['config']['actual_steps'],training_episodes=data['training_episodes'],training_successes=data['training_successes'],bc_updates=data['bc_updates'],checkpoint_loaded=data['checkpoint_loaded'],final_physics=data['after']['diagnostics'])
   else:r['progress']={k:v for k,v in data.items() if k!='physical'}
 rows.append(r)
 if job['status']!='running':
  target=evidence/job['name'];target.mkdir(exist_ok=True)
  for name in ('execution.json','effective_config.json','initial_physics.json','report.json','failure.txt'):
   if (folder/name).exists():__import__('shutil').copy2(folder/name,target/name)
  curve=folder/'learning_curve.jsonl'
  if curve.exists():
   compact=[]
   for line in curve.read_text().splitlines():
    item=json.loads(line);physical=item.pop('physical',{});item['physical_summary']={k:v for k,v in physical.items() if k!='anomalies'};item['anomaly_count']=len(physical.get('anomalies',[]));compact.append(json.dumps(item))
   (target/'learning_curve.jsonl').write_text('\n'.join(compact)+'\n')
complete=sum(x['status']=='completed' for x in rows);failed=sum(x['status']=='failed' for x in rows)
summary=dict(expected_runs=9,completed=complete,failed=failed,rows=rows)
(b/'results.json').write_text(json.dumps(summary,indent=2))
lines=['# DERL 对标纯 PPO 验证结果','',f'已完成 {complete}/9，失败 {failed}。未完成运行不计为0成功。','',
       '| 形态/seed | 状态 | 最终完整回合成功 | 训练步数 |','|---|---|---|---|']
for r in rows:
 score=f"{r['successes']}/{r['episodes']}" if 'successes' in r else '待完成'
 lines.append(f"| {r['name']} | {r['status']} | {score} | {r.get('frames',r.get('progress',{}).get('frames','—'))} |")
lines+=['','## 判读','', '只有实际完整回合成功才能证明本次场景下完成过任务。稳定收敛需结合三个seed的成功率与学习曲线，不能以个别成功或训练回报替代。',
        '若固定预算内没有成功，只能说明本配置/预算下未学会；不能证明纯PPO或该结构不可行。',
        '评估从物理有效预抓取姿态开始；没有BC或父策略载入。最终均值策略与训练随机策略分别报告。',
        '实验采用同一验证扰动集合选择best；不是完全独立的最终测试集。']
if complete+failed==9:
 successful=[r for r in rows if r.get('successes',0)>0]
 lines+=['',f"本轮有 {len(successful)} 次运行在最终评估完成至少一个完整任务。",'']
(b/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('REPORT',complete,failed)
