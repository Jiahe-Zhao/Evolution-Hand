
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from pathlib import Path
import os,xml.etree.ElementTree as E
p=Path(__file__).parent
for finger in ['thumb','index']:
 os.environ.update(RUN_TAG='setup_'+finger,MASS_G='50',CAL_META=str(p/f'assets/calibration_{finger}.json'),OBJECT_XML=str(p/f'assets/calibration_{finger}.xml'))
 ns={'__file__':str(p/'run_pad_calibration.py')};s=(p/'run_pad_calibration.py').read_text();exec(s[:s.index('for it in range')],ns)
 h=ns['h'];np=ns['np'];R=ns['R'];meta=ns['meta'];cp=ns['cp'];axis=np.array(meta['axis']);start=ns['start'];bid=ns['bid']
 if finger=='index':points=cp['particle_q'][:h.real_count][h.regions[:h.real_count]==3]
 else:
  m=h.rigid.mj_model;gid=next(i for i in range(m.ngeom) if '/link_1_2/skin_1_2_' in m.geom(i).name);mesh=m.geom_dataid[gid];verts=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]];local=R.from_quat(m.geom_quat[gid][[1,2,3,0]]).apply(verts)+m.geom_pos[gid];pose=cp['body_q'][bid];points=pose[:3]+R.from_quat(pose[3:]).apply(local)
 support=float(np.max((points-start)@axis));center=start+axis*(support+.0013);meta['center']=center.tolist();meta['surface_support_mm']=support*1000;meta['initial_gap_mm']=.3
 tree=_evo_parse_xml(p/f'assets/calibration_{finger}.xml');tree.find('.//body[@name="item"]').set('pos',' '.join(map(str,center)));tree.write(p/f'assets/calibration_{finger}_v2.xml',encoding='utf8',xml_declaration=True);(p/f'assets/calibration_{finger}_v2.json').write_text(ns['json'].dumps(meta,indent=2));print('SURFACE',finger,meta,flush=True)
