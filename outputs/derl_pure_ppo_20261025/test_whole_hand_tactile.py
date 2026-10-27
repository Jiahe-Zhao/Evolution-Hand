"""Coverage aggregation and reset semantics independent of the simulator."""
import importlib.util,sys,types,unittest
from pathlib import Path
import torch
# Load the sensor with a quaternion identity substitute; these tests concern
# grouping/reset only, not quaternion rotation or PhysX contact validity.
class TactileTests(unittest.TestCase):
 def setUp(self):
  name='isaaclab.utils.math';old=sys.modules.get(name)
  sys.modules[name]=types.SimpleNamespace(quat_apply_inverse=lambda q,v:v)
  try:
   spec=importlib.util.spec_from_file_location('tactile_under_test',Path(__file__).with_name('whole_hand_tactile.py'))
   module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  finally:
   if old is None:sys.modules.pop(name,None)
   else:sys.modules[name]=old
  self.t=module.WholeHandTactile.__new__(module.WholeHandTactile)
  t=self.t;t.groups=[0,1,1,2];t.digits=[1,2];t.force=torch.zeros(2,4,1,3);t.total=torch.zeros(2,4,3)
  t.age=torch.ones(2,4,1);t.previous=t.force.clone();t.delta=t.force.clone();t.stamp=4;t.update=lambda:None
 def test_proximal_contact_counts_but_palm_is_not_a_digit(self):
  t=self.t;t.force[0,:,0,0]=torch.tensor([100.,1.2,.1,2.])
  self.assertTrue(torch.allclose(t.digit_forces()[0],torch.tensor([1.2,2.])))
 def test_more_links_do_not_inflate_threshold(self):
  t=self.t;t.force[0,1:3,0,0]=.6
  self.assertAlmostEqual(t.digit_forces()[0,0].item(),.6,places=5)
 def test_same_side_push_is_not_opposition(self):
  t=self.t;t.force[0,1,0,0]=2;t.force[0,3,0,0]=3
  self.assertGreater(float(t.opposition_cosine()[0]),0)
  t.force[0,3,0,0]=-3
  self.assertLess(float(t.opposition_cosine()[0]),0)
 def test_partial_reset_clears_history_only_selected_env(self):
  t=self.t;t.force.fill_(2);t.reset(torch.tensor([0]))
  self.assertEqual(float(t.age[0].sum()),0);self.assertEqual(float(t.force[0].sum()),0)
  self.assertGreater(float(t.force[1].sum()),0);self.assertIsNone(t.stamp)
if __name__=='__main__':unittest.main()
