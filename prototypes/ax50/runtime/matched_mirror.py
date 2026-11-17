import numpy as np
import xml.etree.ElementTree as E
from scipy.spatial.transform import Rotation as R
L=np.diag([-1.,1.,1.]);W=np.diag([1.,1.,-1.])
def mirrored(root):
 fmt=lambda x:' '.join('%.15g'%t for t in x)
 def origin(el,pa,ch):
  o=el.find('origin')
  if o is None:o=E.SubElement(el,'origin')
  q=R.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
  o.set('xyz',fmt(pa@np.fromstring(o.get('xyz','0 0 0'),sep=' ')));o.set('rpy',fmt(R.from_matrix(pa@q@ch).as_euler('xyz')))
 children={j.find('child').get('link') for j in root.findall('joint')};basenames={l.get('name') for l in root.findall('link')}-children
 for j in root.findall('joint'):
  pa=W if j.find('parent').get('link') in basenames else L;origin(j,pa,L);ax=j.find('axis')
  if ax is not None:
   v=L@np.fromstring(ax.get('xyz'),sep=' ');ax.set('xyz',fmt(-v if j.get('type') in ('revolute','continuous') else v))
 for l in root.findall('link'):
  pa=W if l.get('name') in basenames else L
  for el in list(l):
   if el.tag in ('visual','collision','inertial'):origin(el,pa,L)
   if el.tag=='inertial':
    a=el.find('inertia')
    for key in ('ixy','ixz'):a.set(key,str(-float(a.get(key,'0'))))
