from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET
NAMES=['diagnostic_wrist_x','diagnostic_wrist_y','diagnostic_wrist_z','diagnostic_wrist_roll','diagnostic_wrist_pitch','diagnostic_wrist_yaw']
def build(source,destination):
 source=Path(source).resolve(); destination=Path(destination).resolve()
 if destination.exists(): raise FileExistsError(destination)
 original=source.read_bytes(); root=ET.fromstring(original)
 children={j.find('child').get('link') for j in root.findall('joint')}; roots=[l.get('name') for l in root.findall('link') if l.get('name') not in children]
 if len(roots)!=1: raise ValueError('Expected one hand root')
 destination.mkdir(parents=True); asset_root=destination/'assets'; shutil.copytree(source.parent.parent,asset_root)
 for name in ('diagnostic_world','diagnostic_stage_x','diagnostic_stage_y','diagnostic_stage_z','diagnostic_stage_roll','diagnostic_stage_pitch'):
  link=ET.SubElement(root,'link',name=name); inertial=ET.SubElement(link,'inertial'); ET.SubElement(inertial,'mass',value='0.05'); ET.SubElement(inertial,'inertia',ixx='0.00001',iyy='0.00001',izz='0.00001',ixy='0',ixz='0',iyz='0')
 parents=['diagnostic_world','diagnostic_stage_x','diagnostic_stage_y','diagnostic_stage_z','diagnostic_stage_roll','diagnostic_stage_pitch']; descendants=['diagnostic_stage_x','diagnostic_stage_y','diagnostic_stage_z','diagnostic_stage_roll','diagnostic_stage_pitch',roots[0]]
 specs=[('1 0 0','prismatic',-0.25,0.25,100,0.02),('0 1 0','prismatic',-0.25,0.25,100,0.02),('0 0 1','prismatic',-0.25,0.25,100,0.02),('1 0 0','revolute',-1.5708,1.5708,20,2.0),('0 1 0','revolute',-1.5708,1.5708,20,2.0),('0 0 1','revolute',-1.5708,1.5708,20,2.0)]
 for i,name in enumerate(NAMES):
  axis,jtype,lo,hi,eff,vel=specs[i]; j=ET.SubElement(root,'joint',name=name,type=jtype); ET.SubElement(j,'parent',link=parents[i]); ET.SubElement(j,'child',link=descendants[i]); ET.SubElement(j,'origin',xyz='0 0 0',rpy='0 0 0'); ET.SubElement(j,'axis',xyz=axis); ET.SubElement(j,'limit',lower=str(lo),upper=str(hi),effort=str(eff),velocity=str(vel))
 path=asset_root/'urdf/current_agent.urdf'; ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
 report={'source':str(source),'source_sha256':hashlib.sha256(original).hexdigest(),'diagnostic_urdf':str(path),'wrist_joint_names':NAMES,'morphology_genes_include_wrist':False,'axes_frame':'original hand root local frame','added_moving_mass_kg':0.3,'translation_limits_m':[-.25,.25],'rotation_limits_rad':[-1.5708,1.5708],'rotation_effort_Nm':20,'rotation_velocity_rad_s':2.0}
 (destination/'wrist_contract.json').write_text(json.dumps(report,indent=2)); return path,report
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();print(json.dumps(build(a.source,a.destination)[1],indent=2))
