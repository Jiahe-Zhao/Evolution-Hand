"""Generate an independent physical two-jaw bench from exact fingertip meshes."""
import copy,xml.etree.ElementTree as E
from pathlib import Path

def build(source,out):
 source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 original=E.parse(source).getroot();tip=original.find("link[@name='link_2_3']")
 r=E.Element('robot',name='baseline_contact_bench');world=E.SubElement(r,'link',name='base')
 inert=E.SubElement(world,'inertial');E.SubElement(inert,'mass',value='1');E.SubElement(inert,'inertia',ixx='.01',iyy='.01',izz='.01',ixy='0',ixz='0',iyz='0')
 for side,sign in [('left',1),('right',-1)]:
  link=copy.deepcopy(tip);link.set('name',side)
  for mesh in link.findall('.//mesh'):mesh.set('filename',str((source.parent/mesh.get('filename')).resolve()))
  # Explicit test carriage, not a proposed hand actuator mass.
  inert=link.find('inertial');inert.find('mass').set('value',str(float(inert.find('mass').get('value'))+.02))
  for key in ['ixx','iyy','izz']:inert.find('inertia').set(key,str(float(inert.find('inertia').get(key))+1e-6))
  r.append(link);j=E.SubElement(r,'joint',name=side+'_drive',type='prismatic');E.SubElement(j,'parent',link='base');E.SubElement(j,'child',link=side)
  E.SubElement(j,'origin',xyz=f'0 {-sign*.017} 0.1425',rpy='0 0 0');E.SubElement(j,'axis',xyz=f'0 {sign} 0');E.SubElement(j,'limit',lower='0',upper='.012',effort='10',velocity='.01')
 path=out/'bench.urdf';E.ElementTree(r).write(path,encoding='utf-8',xml_declaration=True);return path
