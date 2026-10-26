import json,sys,shutil,statistics
from pathlib import Path
root=Path(sys.argv[1]);state=json.loads((root/'status.json').read_text());rows=[]
for s in state:
 p=root/s['name'];row=dict(name=s['name'],status=s['status'],original_mean_successes=s.get('original_mean_successes'))
 if (p/'summary.json').exists():
  trials=json.loads((p/'summary.json').read_text());row['trials']=[{k:v for k,v in t.items() if k not in ('passed','final_hold','diagnostics','curve')}|{'physics':{k:v for k,v in t['diagnostics'].items() if k!='anomalies'}} for t in trials]
  samples=[t['successes']/t['num_envs'] for t in trials if t['action_mode']=='sample'];row['sample_mean']=statistics.mean(samples) if samples else None;row['sample_range']=[min(samples),max(samples)] if samples else None
  evidence=root/'evidence'/s['name'];evidence.mkdir(parents=True,exist_ok=True)
  for f in ('manifest.json','summary.json','completed.json','failure.txt'):
   if (p/f).exists():shutil.copyfile(p/f,evidence/f)
 rows.append(row)
(root/'comparison.json').write_text(json.dumps(rows,indent=2))
lines=['# 9个最终检查点复评','',f'完成 {sum(x["status"]=="completed" for x in rows)}/9；失败 {sum(x["status"]=="failed" for x in rows)}。未完成不计为0分。','', '|检查点|状态|新均值动作|随机种子1|随机种子2|随机种子3|随机平均|','|---|---|---|---|---|---|---|']
for r in rows:
 t=r.get('trials',[]);v=[f"{x['successes']}/{x['num_envs']}" for x in t];v+=['—']*(4-len(v));avg=f"{100*r['sample_mean']:.2f}%" if r.get('sample_mean') is not None else '—';lines.append('|'+ '|'.join([r['name'],r['status'],*v,avg])+'|')
lines+=['','每检查点1次均值测试+3个动作随机种子，每次256个25秒回合。相同初始扰动种子1062601。动作种子1062602/1062603/1062604；检查点与观测统计固定，无训练。相同初态复用不等于768个独立初态；随机平均只描述这批扰动与3个噪声种子。','原始训练均值成绩使用旧初态集合；新均值成绩为配对比较依据。模型/碰撞/任务不变；任何曾跌落回合均失败，后续自动重置不算自主恢复。仅当前预抓取附近的抓持抗扰，不代表四任务成功。']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
