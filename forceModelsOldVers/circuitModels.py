import numpy as np
import sys
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import timeit

def circuitModelLinCap(i,t,R,C0,Cox,Cstern,frequency,delta_phi0,*args):
#    delta_phi=delta_phi0*np.cos(frequency*t)
#    dphidt=-delta_phi0*frequency*np.sin(frequency*t)
    delta_phi=delta_phi0*np.sin(frequency*t)
    dphidt=delta_phi0*frequency*np.cos(frequency*t)
    dqdVedl=C0
    dqdVox=Cox
    dqdVstern=Cstern
    dq=i[1]
    if dqdVedl==0:
        dVedl=0
    else:
        dVedl=dq/(dqdVedl) 
    if dqdVox==0:
        dVox=0
    else:
        dVox=dq/(dqdVox)
    if dqdVstern==0:
        dVstern=0
    else:
        dVstern=dq/(dqdVstern)  
    di=(dphidt-2*(dVedl+dVox+dVstern))/R
    dVBulk=di*R
    return [dq, di, dVedl, dVox, dVstern, dVBulk]

    
def circuitModelNonLinCap(i,t,R,C0,Cox,Cstern,frequency,delta_phi0,*args):
    delta_phi=delta_phi0*np.cos(frequency*t)
    dphidt=-delta_phi0*frequency*np.sin(frequency*t)
    #C0=C0/eps
    #delta_phi=delta_phi0*np.sin(frequency*t)
    #dphidt=delta_phi0*frequency*np.cos(frequency*t)
    dqdVedl=C0*np.cosh(i[2]/2)
    dqdVox=Cox
    dqdVstern=Cstern
    dq=i[1]
    if dqdVedl==0:
        dVedl=0
    else:
        dVedl=dq/(dqdVedl) 
    if dqdVox==0:
        dVox=0
    else:
        dVox=dq/(dqdVox)
    if dqdVstern==0:
        dVstern=0
    else:
        dVstern=dq/(dqdVstern)  
    
    di=(dphidt-2*(dVedl+dVox+dVstern))/R
    dVBulk=di*R
    return [dq, di, dVedl, dVox, dVstern, dVBulk]      
      
def circuitModelNonLinCapVarRes(i,t,C0,Cox,Cstern,frequency,delta_phi0,*args):
    #delta_phi=delta_phi0*np.cos(frequency*t)
    #dphidt=-delta_phi0*frequency*np.sin(frequency*t)
    delta_phi=delta_phi0*np.sin(frequency*t)
    dphidt=delta_phi0*frequency*np.cos(frequency*t)
    
    dqdVedl=C0*np.cosh(i[2]/2)
    dqdVox=Cox
    dqdVstern=Cstern
    R=(i[6])
    dq=i[1]
    if dqdVedl==0:
        dVedl=0
    else:
        dVedl=dq/(dqdVedl) 
    if dqdVox==0:
        dVox=0
    else:
        dVox=dq/(dqdVox)
    if dqdVstern==0:
        dVstern=0
    else:
        dVstern=dq/(dqdVstern) 
    
    dC=(-2*dq)*np.sign(i[0])
    dR=(-1.0/(2.0*(i[5]**2)))*dC
    di=(dphidt-2*(dVedl+dVox+dVstern)-dR*dq)/R
    dVBulk=di*R+i[1]*dR 
    return [dq, di, dVedl, dVox, dVstern, dC, dR, dVBulk]  
 
def circuitModelNonLinCapROxVarRes(i,t,C0,Cox,Cstern,ROx,frequency,delta_phi0,*args):
    q = i[0];   I = i[1];   vEDL = i[2];
    vOx = i[3]; vStern = i[4];  c = i[5];
    ROx = i[6]; R = i[7];   vBulk = i[8];
    #delta_phi=delta_phi0*np.cos(frequency*t)
    #dphidt=-delta_phi0*frequency*np.sin(frequency*t)
    delta_phi=delta_phi0*np.sin(frequency*t)
    dphidt=delta_phi0*frequency*np.cos(frequency*t)
    
    dqdVedl=C0*np.cosh(vEDL/2)
    dqdVstern=Cstern
    dq=I
    dVox = (I-vOx/ROx)/Cox;
    if dqdVedl==0:
        dVedl=0
    else:
        dVedl=dq/(dqdVedl) 
    if dqdVstern==0:
        dVstern=0
    else:
        dVstern=dq/(dqdVstern) 
    
    dC=(-2*dq)*np.sign(i[0]);
    dR=(-1.0/(2.0*(c**2)))*dC;
    di=(dphidt-2*(dVedl+dVox+dVstern)-dR*dq)/R;
    dVBulk=di*R+I*dR; 
    return [dq, di, dVedl, dVox, dVstern, dC, dROx, dR, dVBulk]  

       
def circuitModel_NonLinCap_InhomResPython(i,t,dx,eps,C0,Cox,Cstern,frequency,delta_phi0,A,b,*args): 
    #delta_phi=delta_phi0*np.cos(frequency*t)
    #dphidt=-delta_phi0*frequency*np.sin(frequency*t)
    delta_phi=delta_phi0*np.sin(frequency*t)
    dphidt=delta_phi0*frequency*np.cos(frequency*t)
    
    dqdVedl=C0*np.cosh(i[2]/2)
    dqdVox=Cox
    dqdVstern=Cstern
    
    qInner=-2*np.sinh(i[2]/2)
    dq=i[1]
    if dqdVedl==0:
        dVedl=0
    else:
        dVedl=dq/(dqdVedl) 
    if dqdVox==0:
        dVox=0
    else:
        dVox=dq/(dqdVox)
    if dqdVstern==0:
        dVStern=0
    else:
        dVStern=dq/(dqdVstern) 
    
    dw0=(-1.0/eps)*(qInner*(eps*dVedl))
    
    cInit=2.0
    c=np.reshape(i[6:-1],(1./dx,1))
    
    c1=(c-cInit)/eps
    dc1=A*c1+b*dw0/dx
    dc=np.array(eps*dc1)    
    R=1.0/c
    
    R=(0.5*(R[0]+R[-1])+np.sum(R[1:-1]))*dx  
    dR=np.multiply(-np.square(1.0/c),dc)
    dR=(0.5*(dR[0]+dR[-1])+np.sum(dR[1:-1]))*dx
    
    di=(dphidt-2*(dVedl+dVox+dVStern)-dq*dR)/R
    dVBulk=(di*R+i[1]*dR)

    du=np.append(dq,di)#([0,delta_phi0/R,0,R,cInit,eps,delta_phi0])
    du=np.append(du,dVedl)
    du=np.append(du,dVox)
    du=np.append(du,dVStern)
    du=np.append(du,dR)
    du=np.append(du,dc)
    du=np.append(du,dVBulk)
    return du# [dq, di, dVedl, dR, dc, deps, dVR]
 