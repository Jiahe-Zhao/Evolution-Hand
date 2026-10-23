"""Verify the baseline and all sixteen single-gene neighbors in actual URDFs."""
import contextlib
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from build_candidate import build
from genes import NAMES

BASE=Path(__file__).resolve().parent


def signature(path):
    root=ET.parse(path).getroot()
    return [(j.get('name'),j.get('type'),j.find('limit').attrib if j.find('limit') is not None else {}) for j in root.findall('joint')]


def main():
    old=Path('/home/zjh/Evolution_PC/outputs/motion_collision_revision_20261021/matrix/baseline/right/urdf/current_agent.urdf')
    baseline=BASE/'checks/baseline/right/urdf/current_agent.urdf'
    assert baseline.read_bytes()==old.read_bytes(), 'Baseline URDF changed'
    # URDF equality alone would miss changes in referenced mesh contents.
    for ref in ET.parse(old).getroot().findall('.//mesh'):
        filename=ref.get('filename')
        assert (old.parent/filename).read_bytes()==(baseline.parent/filename).read_bytes(), filename
    rows=[dict(name='baseline',passed=True,urdf_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),meshes_identical=True)]
    for i,name in enumerate(NAMES):
        for level in [-1,1]:
            label=f'{name}_{level:+d}'; output=BASE/'matrix'/label
            try:
                code=[0]*8;code[i]=level
                with (BASE/'matrix_build.log').open('a') as log,contextlib.redirect_stdout(log):
                    result=build(BASE/'baseline_genotype.json',code,output,'/home/zjh/Evolution_PC/Isaaclab_other')
                right=Path(result['artifacts']['right'])
                assert signature(right)==signature(baseline),'Joint interface changed'
                assert right.read_bytes()!=baseline.read_bytes(),'Gene has no mechanical effect'
                rows.append(dict(name=label,passed=result['passed'],checks=result['checks'],joint_interface_unchanged=True))
            except Exception as error:rows.append(dict(name=label,passed=False,error=f'{type(error).__name__}: {error}'))
            (BASE/'matrix_results.json').write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
    return rows


if __name__=='__main__':main()
