import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from genes import BASELINE, decode, mutate, validate
from inheritance import load_actor, validate_request, check_seed_quality
from evolution import offspring, select

HERE = Path(__file__).resolve().parent


class GenesTest(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads((HERE/'baseline_genotype.json').read_text())

    def test_identity_and_immutable_baseline(self):
        original = copy.deepcopy(self.baseline)
        middle, _ = decode(self.baseline, BASELINE, .062)
        self.assertEqual(middle, original)
        for i in range(8):
            for level in [-1, 1]:
                code = [0]*8; code[i] = level
                first = decode(self.baseline, code, .062)
                self.assertEqual(first, decode(self.baseline, code, .062))
                self.assertEqual(self.baseline, original)

    def test_total_span_and_proximal_fraction(self):
        def spans(hand, d):
            links = {l['name_code']:l for l in hand['links']}
            return [sum(v*v for v in links[f'link_{d}_{i}']['joint_origin_translation'])**.5 for i in range(1,4)]+[links[f'link_{d}_3']['geometry_length']]
        hand, _ = decode(self.baseline, [0,0,1,1,-1,0,0,0], .062)
        for d in range(1,6):
            before, after = spans(self.baseline,d), spans(hand,d)
            self.assertAlmostEqual(sum(after)/sum(before), .9 if d==1 else 1.1)
            if d!=1:self.assertAlmostEqual(after[0]/sum(after)-before[0]/sum(before), .05)

    def test_mutations_are_adjacent_and_bounded(self):
        rng=random.Random(9); code=BASELINE
        for _ in range(1000):
            child=mutate(code,rng); validate(child)
            self.assertEqual(sum(abs(a-b) for a,b in zip(child,code)),1); code=child
        for code in [[0]*7,[True]*8,[.0]*8,[2]*8]:
            with self.assertRaises(ValueError):validate(code)

    def test_bc_generation_guard(self):
        self.assertTrue(validate_request(0))
        self.assertFalse(validate_request(1,checkpoint='parent'))
        for arguments in [dict(generation=1),dict(generation=1,checkpoint='p',refit=True),
                          dict(generation=1,checkpoint='p',mode='bc'),dict(generation=1,checkpoint='p',collect=True),
                          dict(generation=2,checkpoint='p',residual_bc=True),dict(generation=2,checkpoint='p',bc_anchor=.1)]:
            with self.assertRaises(ValueError):validate_request(**arguments)

    def test_lineage_and_selection(self):
        parents=[dict(id='baseline',code=list(BASELINE),generation=0,checkpoint='parent.pth')]
        rows=offspring(parents,8,31)
        self.assertEqual(len({tuple(r['code']) for r in rows}),8)
        self.assertTrue(all(r['generation']==1 and not r['bc_allowed'] and r['parent_id']=='baseline' for r in rows))
        for i,r in enumerate(rows):r.update(evaluation_contract='same',evaluation_seed=12,episodes=64,training_budget=2000000,successes=i)
        chosen=select(rows,4);self.assertEqual(len(chosen),4);self.assertEqual(chosen[0]['successes'],7)
        rows[0]['evaluation_seed']=2
        with self.assertRaises(ValueError):select(rows,4)

    def test_descendants_can_learn_from_initial_failure(self):
        check_seed_quality(1,0,64)
        check_seed_quality(10,0,64)
        check_seed_quality(0,63,64)
        with self.assertRaises(ValueError):check_seed_quality(0,0,64)


class InheritanceTest(unittest.TestCase):
    def test_only_actor_transfers(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.a2c_network=torch.nn.Module()
                self.a2c_network.actor_mlp=torch.nn.Sequential(torch.nn.Linear(3,4))
                self.a2c_network.mu=torch.nn.Linear(4,2)
                self.a2c_network.critic_mlp=torch.nn.Linear(3,4)
                self.a2c_network.value=torch.nn.Linear(4,1)
                self.a2c_network.sigma=torch.nn.Parameter(torch.zeros(2))
        parent=Model(); child=Model()
        with torch.no_grad():
            for p in parent.parameters():p.fill_(7.)
        before={k:v.clone() for k,v in child.state_dict().items()}
        agent=SimpleNamespace(model=child,optimizer=torch.optim.Adam(child.parameters()),ppo_device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'parent.pth';torch.save(dict(model=parent.state_dict(),optimizer={'old':'do not load'},epoch=999),path)
            report=load_actor(agent,path,{'same':True},{'same':True})
            for k,v in child.state_dict().items():
                torch.testing.assert_close(v,parent.state_dict()[k] if k in report['copied_keys'] else before[k],rtol=0,atol=0)
            self.assertFalse(agent.optimizer.state)
            with self.assertRaises(ValueError):load_actor(agent,path,{'task':1},{'task':2})


if __name__=='__main__':unittest.main()
