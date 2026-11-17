#include <math.h>
#include <string.h>
void step(int n,int m,const double *lo,const unsigned char *v,double *w,double *f){
double g=.13,k=.021*.75*.75;
for(int it=0;it<100;it++){double mx=0;for(int p=0;p<2;p++)for(int i=1;i<n-1;i++)for(int j=1+(p+i)%2;j<m-1;j+=2){int h=i*m+j;if(!v[h])continue;double q=fmax(lo[h],g*(w[h+1]+w[h-1]+w[h+m]+w[h-m])/(k+4*g));mx=fmax(mx,fabs(q-w[h]));w[h]=q;}if(mx<.00003)break;}
for(int i=1;i<n-1;i++)for(int j=1;j<m-1;j++){int h=i*m+j;if(v[h]&&lo[h]>0&&w[h]<=lo[h]+.00001)f[h]=fmax(0,(k+4*g)*w[h]-g*(w[h+1]+w[h-1]+w[h+m]+w[h-m]));}}
