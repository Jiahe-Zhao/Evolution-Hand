from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
O=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_pebble07');a=np.load(O/'topwide_mesh_preview.npz');fig=plt.figure(figsize=(12,8));light=np.array([-.4,-.7,.7]);light/=np.linalg.norm(light)
for j,(v,f,label) in enumerate([(a['original'],ConvexHull(a['original']).simplices,'Original'),(a['vertices'],a['faces'],'Rounded, mildly top-wide')]):
 v=v*1000;ax=fig.add_subplot(2,2,j+1,projection='3d');tri=v[f];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None];cent=tri.mean(1)-v.mean(0);norm*=np.where((norm*cent).sum(1)<0,-1,1)[:,None];shade=.60+.35*np.maximum(norm@light,0);colors=np.c_[shade[:,None]*np.array([.68,.47,.27]),np.ones(len(f))];ax.add_collection3d(Poly3DCollection(tri,facecolors=colors,edgecolors='none'));ax.set_xlim(-11,11);ax.set_ylim(-9,9);ax.set_zlim(-5,7);ax.set_box_aspect((22,18,12));ax.view_init(18,-58);ax.set_title(label);ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)')
 ax=fig.add_subplot(2,2,j+3);pts=v[:,[1,2]];h=ConvexHull(pts);q=pts[h.vertices];q=np.vstack([q,q[0]]);ax.fill(q[:,0],q[:,1],color='#af8050',alpha=.65);ax.plot(q[:,0],q[:,1],color='#654523');ax.set_xlim(-9,9);ax.set_ylim(-5,7);ax.set_aspect('equal');ax.grid(alpha=.25);ax.set_title(label+' — side silhouette');ax.set_xlabel('Y (mm)');ax.set_ylabel('Z (mm)')
fig.suptitle('Target stone geometry comparison | same bounds and 2.110 g assembly mass');fig.tight_layout();fig.savefig(O/'topwide_geometry_comparison.png',dpi=150);print(O/'topwide_geometry_comparison.png')
