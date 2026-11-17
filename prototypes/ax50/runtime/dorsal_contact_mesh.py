import numpy as np
from scipy.spatial import ConvexHull
def shell(length,core,seg):
 R=core+.003;delta=.00085 if seg else .00145
 points=[]
 for end in [0,1]:
  for lat in np.linspace(0,np.pi/2,7):
   for th in np.arange(48)*np.pi/24:
    x=np.cos(th)*R*np.cos(lat);y=np.sin(th)*R*np.cos(lat)
    # The palmar hemisphere is unchanged, dorsal hemisphere compressed smoothly.
    if y<0:y*=1-(delta/R)*(-np.sin(th))**2
    points.append([x,y,end*length+(1 if end else -1)*R*np.sin(lat)])
 v=np.unique(np.round(points,10),axis=0);h=ConvexHull(v);f=h.simplices.copy();flip=(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])*h.equations[:,:3]).sum(1)<0;f[flip]=f[flip,::-1];return v.astype('f'),f.flatten()
