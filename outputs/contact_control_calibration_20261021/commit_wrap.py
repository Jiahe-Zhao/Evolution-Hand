"""Commit only this grasp change and its compact evidence."""
import subprocess
from pathlib import Path
base=Path(__file__).resolve().parent
repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=base,text=True).strip())
paragraph='\n## 包覆抓握后续修改\n\n见 [WRAP_GRASP.md](WRAP_GRASP.md)。仅修改抓握规划与闭合反馈，物理和驱动参数保持不变。`wrap04` 在固定初始条件下通过完整操作测试，末端滑移 0.336 mm；四指与拇指均承重，但中节接触仍不均匀。此结果不替代随机初始条件、时间步和变异手验证，训练仍暂停。\n'
for name in ['README.md','RESULTS.md']:
 path=base/name
 if '## 包覆抓握后续修改' not in path.read_text():path.write_text(path.read_text()+paragraph)
names=['plan_long_side.py','plan_long_side_contact.py','plan_long_side_balanced.py','plan_wrap.py','plan_wrap_cover.py','calibrate_long_side.py','calibrate_object_follow_trial.py','calibrate_wrap.py','check_long_side.py','summarize_wrap.py','run_wrap_validation.py','package_wrap.py','commit_wrap.py','WRAP_GRASP.md','README.md','RESULTS.md','wrap_summary.json','long_side_checks.json','wrap_evidence_manifest.json']
files=[base/name for name in names]
files+=list(base.glob('plans_long_side*.json'))+list(base.glob('plans_wrap*.json'))
for name in ['long_side01','long_side02','long_side03','long_side04','long_side05','wrap01','wrap02','wrap03','wrap04']:
 for leaf in ['report.json','controller_snapshot.py']:
  files.append(base/name/leaf)
files+=[base/'wrap04/audit.json',base/'long_side04/audit.json']
paths=[str(path.relative_to(repo)) for path in files]
assert all(path.is_file() for path in files)
subprocess.run(['git','add','-f','--',*paths],cwd=repo,check=True)
subprocess.run(['git','diff','--cached','--stat','--',*paths],cwd=repo,check=True)
subprocess.run(['git','commit','-m','Implement opposed wrapping grasp with four-finger contact before lift','--',*paths],cwd=repo,check=True)
print('COMMIT',subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
