"""Summarize saved evidence; this does not run training."""
import json
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent
run=BASE/'smoke_g1c'
initial=torch.load(run/'inherited_initial.pth',map_location='cpu',weights_only=False)
warm=torch.load(run/'critic_ready.pth',map_location='cpu',weights_only=False)
final=torch.load(run/'ppo_final.pth',map_location='cpu',weights_only=False)
keys=[k for k in initial['model'] if k.startswith(('a2c_network.actor_mlp.','a2c_network.mu.'))]
def delta(other):return max(float((initial['model'][k]-other['model'][k]).abs().max()) for k in keys)
report=json.loads((run/'report.json').read_text())
inheritance=json.loads((run/'inheritance.json').read_text())
warm_report=json.loads((run/'critic_warmup.json').read_text())
assert report['generation']==1 and report['bc_updates']==0 and not report['bc_allowed']
assert not (run/'bc_fit.json').exists() and not (run/'dataset.json').exists()
assert inheritance['actor_max_copy_error']==0 and delta(warm)==0 and delta(final)>0
assert not initial['optimizer']['state']
matrix=json.loads((BASE/'matrix_results.json').read_text())
summary=dict(status='implementation_smoke_passed',model_checks_passed=sum(r['passed'] for r in matrix),
             model_checks_total=len(matrix),generation=1,code=[0,0,0,0,0,0,0,-1],
             num_envs=4,ppo_epochs=2,horizon=16,training_transitions=128,
             bc_updates=report['bc_updates'],bc_dataset_loaded=False,
             inherited_actor_error=inheritance['actor_max_copy_error'],
             initial_optimizer_state_entries=len(initial['optimizer']['state']),
             actor_change_during_critic_warmup=delta(warm),actor_change_after_ppo=delta(final),
             critic_warmup=warm_report,training_diagnostics=json.loads((run/'training_diagnostics.json').read_text()),
             complete_episode_evaluation=False,convergence_demonstrated=False,
             run=str(run),old_production_training='still paused')
(BASE/'validation_summary.json').write_text(json.dumps(summary,indent=2))
(BASE/'test_results.txt').write_text((BASE/'tests.log').read_text())
print(json.dumps(summary,indent=2))
