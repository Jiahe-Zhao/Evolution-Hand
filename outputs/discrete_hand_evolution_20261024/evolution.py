"""Opt-in ternary evolution jobs. Does not resume the legacy evolution process."""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from genes import BASELINE, NAMES, digest, mutate, validate
from inheritance import validate_request, read_contract


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(data, indent=2)); temporary.replace(path)


def offspring(parents, count, seed):
    """Births mutate one adjacent level; parent identity is retained explicitly."""
    if not parents or count < 1:
        raise ValueError('Need parents and a positive population size')
    rng = random.Random(seed); children = []; seen = {tuple(p['code']) for p in parents}
    for _ in range(count*200):
        parent = rng.choice(parents); code = mutate(parent['code'], rng)
        if code in seen: continue
        seen.add(code)
        children.append(dict(id=digest([parent['id'], code, seed])[:16], code=list(code),
                             generation=parent['generation']+1, parent_id=parent['id'],
                             parent_checkpoint=parent['checkpoint'], bc_allowed=False))
        if len(children) == count: return children
    raise ValueError('Neighborhood exhausted; reduce births or retain more diverse parents')


def select(rows, keep):
    """Keep an elite, and reserve one slot for genotype diversity when possible.

    Ranking uses identical held-out trials, not training reward. Do not compare
    candidates with different budgets, task/physics contracts or evaluation seeds.
    """
    if not 1 <= keep <= len(rows): raise ValueError('Invalid survivor count')
    fields = ('evaluation_contract', 'evaluation_seed', 'episodes', 'training_budget')
    if any(any(r[k] != rows[0][k] for k in fields) for r in rows):
        raise ValueError('Candidates must share evaluation conditions and training budget')
    if any(not 0 <= r['successes'] <= r['episodes'] or r['episodes'] <= 0 for r in rows):
        raise ValueError('Invalid success counts')
    ranked = sorted(rows, key=lambda r:(r['successes']/r['episodes'], -r.get('falls', 0)), reverse=True)
    result = ranked[:max(1, keep-1)]
    if keep > 1:
        rest = [r for r in ranked if r not in result]
        # One exploration seat retains structurally different hands that may
        # learn more slowly. All other seats retain measured task performance.
        result.append(max(rest, key=lambda r:(min(sum(a!=b for a,b in zip(r['code'],s['code'])) for s in result), r['successes'])))
    return result


def command(job, output):
    generation = job['generation']; checkpoint = job.get('parent_checkpoint', '')
    mode = job.get('mode', 'ppo')
    validate_request(generation, checkpoint=checkpoint, mode=mode)
    candidate = json.loads(Path(job['candidate']).read_text())
    if validate(candidate['code']) != validate(job['code']):
        raise ValueError('Job and phenotype genes disagree')
    if not candidate['passed']: raise ValueError('Invalid physical geometry')
    if checkpoint: read_contract(checkpoint)
    knobs = dict(num_envs=256, epochs=512, critic_warmup=128, ppo_lr=1e-7,
                 finger_std=.0005, wrist_std=.0001, seed=2027)
    unknown = set(job.get('training', {}))-set(knobs)
    if unknown: raise ValueError(f'Unknown training settings: {unknown}')
    knobs.update(job.get('training', {}))
    if mode == 'ppo' and knobs['epochs'] <= knobs['critic_warmup']:
        raise ValueError('Need actor training after critic warmup')
    args = [sys.executable, str(Path(__file__).with_name('worker.py')), '--generation', str(generation),
            '--source', candidate['artifacts']['right'], '--plans', job['plans'], '--output', str(output),
            '--mode', mode, '--headless', '--device', 'cuda:0', '--pad-face', 'ymixed',
            '--kp', '3', '--kd', '.3', '--contact-k', '1500', '--contact-d', '10',
            '--hand-gravity', '--completion-reward', '--gamma', '.9995',
            '--episode-steps', '1500', '--eval-steps', '1500', '--push']
    for key, value in knobs.items(): args.extend(['--'+key.replace('_', '-'), str(value)])
    if checkpoint: args.extend(['--checkpoint', checkpoint])
    if generation == 0 and not checkpoint:
        if not job.get('dataset'): raise ValueError('Generation zero requires validated morphology-specific demonstrations')
        args.extend(['--extra-data', job['dataset'], '--only-extra'])
    return args


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='action',required=True)
    birth=sub.add_parser('offspring'); birth.add_argument('--parents',required=True); birth.add_argument('--count',type=int,default=7); birth.add_argument('--seed',type=int,default=20261024); birth.add_argument('--output',required=True)
    rank=sub.add_parser('select'); rank.add_argument('--results',required=True); rank.add_argument('--keep',type=int,required=True); rank.add_argument('--output',required=True)
    run=sub.add_parser('run'); run.add_argument('--job',required=True); run.add_argument('--output',required=True); run.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.action=='offspring':save(a.output,offspring(json.loads(Path(a.parents).read_text()),a.count,a.seed))
    elif a.action=='select':save(a.output,select(json.loads(Path(a.results).read_text()),a.keep))
    else:
        job=json.loads(Path(a.job).read_text()); cmd=command(job,a.output)
        if not a.execute:print(json.dumps(cmd,indent=2)); return
        output=Path(a.output)
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.with_suffix('.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((output/'report.json').read_text())
        if job['generation']>0 and report['bc_updates']!=0:raise RuntimeError('BC leaked into a descendant')
        save(output/'lineage.json',job)


if __name__=='__main__':main()
