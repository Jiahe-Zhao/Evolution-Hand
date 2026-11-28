import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_encoded_negative_reward_is_not_positive(self):
        spec=importlib.util.spec_from_file_location('read_results',ROOT/'Isaaclab_other/read_results.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'nn').mkdir()
            for name in ['policy_rew__123.4.pth','policy_rew_-3.0.pth']:(run/'nn'/name).touch()
            self.assertEqual(module.get_reward_from_run_dir(str(run)),-3.0)
            (run/'nn/policy_rew_0.75.pth').touch()
            self.assertEqual(module.get_reward_from_run_dir(str(run)),.75)
    def test_separate_spread_models_have_all_meshes(self):
        models=list((ROOT/'models/legacy_spread_20261128').rglob('*.urdf'))
        self.assertEqual(len(models),2)
        for path in models:
            for node in ET.parse(path).iter('mesh'):
                name=node.get('filename')
                self.assertFalse(Path(name).is_absolute(),name)
                self.assertTrue((path.parent/name).is_file(),str(path.parent/name))

if __name__=='__main__':unittest.main()
