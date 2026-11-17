import numpy as np
import ctypes
# 0.75 mm material nodes; extend the same ellipsoid toward the palm root with a tapered collar.
h=.75;ax=6+np.arange(-38,22)*h;az=-4+np.arange(-19,59)*h;X,Z=np.meshgrid(ax,az)
rad=1-(X/14)**2-(Z/9)**2; rad2=1-((X+2)/12)**2-((Z-8)/14)**2
H1=4*np.sqrt(np.maximum(rad,0));H2=3.4*np.sqrt(np.maximum(rad2,0))-1.4
valid=(rad>0)|((rad2>0)&(H2>0));Y=np.maximum(H1,H2)+np.maximum(0,2-abs(H1-H2))**2/8;Y=np.where(valid,Y,0.)
lib=ctypes.CDLL('./layer.so')
def response(lo,valid,w):
 f=np.zeros_like(w);lib.step(*w.shape,*[ctypes.c_void_p(a.ctypes.data) for a in (lo,valid,w,f)]);return f
