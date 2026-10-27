import unittest
import torch
from goal_contract import LiftGoals

class GoalContractTest(unittest.TestCase):
    def test_goal_changes_observation_reward_and_success(self):
        g=LiftGoals(3,'cpu',1,heights=(.025,.040,.055),tolerance=.002)
        rise=torch.full((3,),.025)
        self.assertEqual(g.features(rise).shape,(3,4))
        self.assertTrue(torch.allclose(g.features(rise)[:,1],(g.target-rise)/.05))
        self.assertEqual(g.reached(rise).tolist(),[True,False,False])
        self.assertGreater(g.lift_reward(rise)[0],g.lift_reward(rise)[2])
        self.assertTrue(torch.all(g.reached(g.target)))
        self.assertTrue(torch.allclose(g.lift_reward(g.target),torch.ones(3)))
        self.assertTrue(torch.all(g.lift_reward(g.target+.05)<.01))

    def test_partial_reset_and_reproducible_training(self):
        a=LiftGoals(8,'cpu',3);b=LiftGoals(8,'cpu',3)
        ids=torch.tensor([1,4,7]);old=a.target.clone()
        a.reset(ids);b.reset(ids)
        self.assertTrue(torch.equal(a.target,b.target))
        self.assertTrue(torch.equal(a.target[[0,2,3,5,6]],old[[0,2,3,5,6]]))
        a.reset(torch.arange(8),evaluation=True)
        self.assertTrue(torch.equal(a.target,a.levels[torch.arange(8)%3]))

    def test_targets_cannot_all_be_met_by_one_height(self):
        g=LiftGoals(3,'cpu',1)
        self.assertGreater(float(g.target.max()-g.tolerance),float(g.target.min()+g.tolerance))

    def test_invalid_goals(self):
        for heights,tol in [((.01,),.002),((.03,),-.01),((.03,),.04)]:
            with self.assertRaises(ValueError):LiftGoals(2,'cpu',1,heights,tol)

if __name__=='__main__':unittest.main()
