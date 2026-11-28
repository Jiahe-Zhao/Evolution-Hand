"""Gravity-only edge impact: verify contact/force readout and reject unheld tool."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
import json
import surface_objects as so
from env import episode
from build import OUT,E
make=so.scene
def fixture(t):
    p=make(t);r=_evo_parse_xml(p);b=r.find("worldbody/body[@name='item']");b.set('pos','-.099 .010 .055');b.set('euler','0 -.524 0')
    r.find('worldbody/body').set('pos','.3 0 .091');dest=OUT/'scrape_probe.xml';r.write(dest);return dest
so.scene=fixture
g=episode('scrape',gpu=True);m=next(g);z=[g.send(m['base'])['task_metrics'] for _ in range(160)]
data=dict(peak_edge_N=max(r['force_N'] for r in z),max_coverage=max(r['coverage'] for r in z),any_grip=any(r['held'] for r in z),success=any(r['success'] for r in z))
print(data);assert data['peak_edge_N']>0 and not data['any_grip'] and data['max_coverage']==0
(OUT/'scrape_probe.json').write_text(json.dumps(data))
