
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
import sys,json,hashlib,zipfile,shutil,os
from pathlib import Path
from xml.etree import ElementTree as E
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));ROOT=P.parents[2];tag=sys.argv[1]
for name in ['ARPHICPL.TXT','hao_reference.json']:shutil.copy2(ROOT/'outputs/nov20/hao14'/name,P/name)
selected=['RESULTS.md','SUMMARY.json','grasp_seed.npz','rebuild_audit.json','rebuild_resolved.mjb','ARPHICPL.TXT','hao_reference.json']
for label in ['verified','control',tag]:
 for suffix in ['.xml','.npz','_operation.mp4','_model.json','_contacts.csv','_trace.csv','_resolved_contacts.json','_replay_audit.json']:
  selected.append(label+suffix)
for name in ['verified_metrics.json','verified_geometry_audit.json','verified_curves.png','verified_preview.jpg',tag+'_writing_metrics.json',tag+'_solver_status.json',tag+'_overlay.png',tag+'_geometry_audit.json',tag+'_preview.jpg',tag+'_resolved.mjb']:
 selected.append(name)
files=[P/name for name in selected]+list((ROOT/'scripts/nov20/brush14').glob('*.py'))+[ROOT/'docs/TASK14_HAO_BRUSH.md',ROOT/'docs/SCENES_20_20261116.md']
assets=set(Path(m.get('file')) for m in _evo_parse_xml(P/'verified.xml').iter('mesh') if m.get('file'))
assets.add(ROOT/'outputs/v5_acceptance/hand_111/left/hand.urdf')
manifest={}
for path in files+sorted(assets):
 if not path.exists():raise FileNotFoundError(path)
 manifest[os.path.relpath(path,P)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2))
public=['RESULTS.md','SUMMARY.json','manifest.json','verified_operation.mp4','control_operation.mp4',tag+'_operation.mp4','verified_curves.png',tag+'_overlay.png','verified_metrics.json',tag+'_writing_metrics.json','verified_geometry_audit.json',tag+'_geometry_audit.json','verified_contacts.csv',tag+'_contacts.csv','verified_trace.csv',tag+'_trace.csv',tag+'_solver_status.json','verified_preview.jpg',tag+'_preview.jpg','ARPHICPL.TXT']
with zipfile.ZipFile(P/'brush14_results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name in public:z.write(P/name,name)
print('packaged',P/'brush14_results.zip')
