#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:33:12 2018

@author: odibua
"""

import numpy as np
import matplotlib as mplib
import matplotlib.pyplot as plt
#import matplotlib.animation as anim
from scipy import integrate
import runModels

def calcTimeConstantOxideOnly(c0NomIn,params):
    c0NomIn=(c0NomIn*1000.0)*params.NA
    #print c0NomIn
    lambda_d=np.sqrt((params.epsilonR*params.epsilon0*params.kB*params.T)/(2*(params.eCharge**2)*c0NomIn)) 
    c0NomIn=c0NomIn/(params.NA*1000)
    eps=lambda_d/params.L
    oxideLayer=1.0
    sternLayer=1.0
    doubleLayer=1.0
            
    #Define Resistance(s) and Capacitance(s)
    C0=2*eps*doubleLayer #Dimensionless linear component electric double layer capacitor
    c0=1 #Dimensionless bulk concentration of single ion
    R=1./(2*c0) #Dimensionless initial resistance of bulk       
    Cox=(params.epsilonOx/params.epsilonR)*(lambda_d/params.lambda_Ox)*C0*oxideLayer  #Dimensionless capacitance of oxide
    Cstern=(params.epsilonStern/params.epsilonR)*(lambda_d/params.lambda_Stern)*C0*sternLayer #Dimensionless capacitance of Stern Layer

    print(Cox)
    tau=(Cox*R)*((params.L**2)/params.Di)
    return tau
    
    
class combDriveModels():            
    def classicCircuitModel(self,VRMS,combDriveParams,params):
        def classicCircuitModelFit(omegaDimIn,tau1,k):           
            k=k*1e-6;
            epsilon0 = params.epsilon0;
            epsilonBulk=params.epsilonR;
            NFingers=combDriveParams.NFingers;
            NCombs=combDriveParams.NCombs;
            g = combDriveParams.d;
            b = combDriveParams.b;
            
            fTau= ((0.5*omegaDimIn*tau1)**2)/(1+((0.5*omegaDimIn*tau1)**2))  
            #print(k,((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g)))
            displ=(1.0/k)*((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g))*fTau*(VRMS**2)  
            
            return displ
        return classicCircuitModelFit
    
    def leakyDielectricModel(self,VRMS,combDriveParams,params):
        def leakyDielectricModelFit(omegaDimIn,ROx,COx,RBulk,k):
            k = k*1e-6;
            epsilonBulk = params.epsilonR;
            epsilon0  = params.epsilon0;
            b = combDriveParams.b;
            g = combDriveParams.d; 
            NFingers = combDriveParams.NFingers;
            NCombs = combDriveParams.NCombs;
            
            #Cox = epsilon0*epsilonOx/(lambdaOx);
            #CBulk = (epsilon0*epsilonBulk)/g;
            #RBulk = tauBulk/CBulk;
            
            ZOx = ROx/(1+omegaDimIn*ROx*COx*1j);
            ZBulk = RBulk#/(1+omegaDimIn*tauBulk*1j);
            Z = 2*ZOx+ZBulk;

            fTau =abs(ZBulk/Z)**2;
            displ=(1.0/k)*((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g))*fTau*(VRMS**2);
                       
            return displ
            
        return leakyDielectricModelFit
    
    def leakyDielectricSternModel(self,VRMS,combDriveParams,params):
        def leakyDielectricSternModelFit(omegaDimIn,ROx,COx,CStern,RBulk,k):
            k=k*1e-6;
            epsilonOx = params.epsilonOx;
            epsilonBulk = params.epsilonR;
            epsilon0  = params.epsilon0;
            b = combDriveParams.b;
            g = combDriveParams.d;
            NFingers = combDriveParams.NFingers;
            NCombs = combDriveParams.NCombs;
            
            #Cox = epsilon0*epsilonOx/(lambdaOx);
            CBulk = (epsilon0*epsilonBulk)/g;
#            CStern = (epsilon0*epsilonBulk)/lambdaStern;
#            CEDL = (epsilon0*epsilonBulk)/lambdaEDL;
            #RBulk = tauBulk/CBulk;
            
            ZOx = ROx/(1+omegaDimIn*ROx*COx*1j);
            ZBulk = RBulk#/(1+omegaDimIn*tauBulk*1j);
            ZStern = 1/(1j*omegaDimIn*CStern);
            #ZEDL = 1/(1j*omegaDimIn*CEDL);
            Z = 2*(ZOx+ZStern)+ZBulk;

            fTau =abs(ZBulk/Z)**2;
            displ=(1.0/k)*((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g))*fTau*(VRMS**2);
            
            
            return displ
            
        return leakyDielectricSternModelFit        
        
    def variableResistorCircuitModel(self,Vpp,params,combDriveParams,model):
#        def variableResistorCircuitModelFit(omegaDimIn,alphaOx,alphaStern,c0NomIn,k):
        def variableResistorCircuitModelFit(omegaDimIn,Cox,Cstern,c0NomIn,k):
            k=k*1e-6;
            run=runModels.implementationClass();
            NFingers = combDriveParams.NFingers;
            NCombs = combDriveParams.NCombs;
            b = combDriveParams.b;
            g = combDriveParams.d;
            epsilonBulk = params.epsilonR;
            epsilon0  = params.epsilon0;
            epsilonOx = params.epsilonOx;
            c0NomIn=c0NomIn*100e-6;
            #print(c0NomIn)
            c0NomIn=(c0NomIn*1000.0)*params.NA
            lambda_d=np.sqrt((params.epsilonR*params.epsilon0*params.kB*params.T)/(2*(params.eCharge**2)*c0NomIn)) 
            c0NomIn=c0NomIn/(params.NA*1000)
            eps=lambda_d/params.L
            #print("eps",eps,c0NomIn)
            oxideLayer=1.0
            sternLayer=1.0
            doubleLayer=1.0
        
            #Define Resistance(s) and Capacitance(s)
            C0=2*eps #Dimensionless linear component electric double layer capacitor
            c0=1 #Dimensionless bulk concentration of single ion
            R=1./(2*c0) #Dimensionless initial resistance of bulk       
#            Cox=alphaOx*C0*oxideLayer  #Dimensionless capacitance of oxide
#            Cstern=alphaStern*C0*sternLayer #Dimensionless capacitance of Stern Layer
            C0=C0*doubleLayer

#            #CBulk=(params.epsilonR/params.epsilonR)*(lambda_d/params.L)*C0*bulkCapacitance
#            Cox=(params.epsilonOx/params.epsilonR)*(lambda_d/params.lambda_Ox)*C0*oxideLayer  #Dimensionless capacitance of oxide
#            Cstern=(params.epsilonStern/params.epsilonR)*(lambda_d/params.lambda_Stern)*C0*sternLayer #Dimensionless capacitance of Stern Layer
#            C0=0;
##            Cox = (params.epsilonOx/params.epsilonR)*(lambda_d/params.L)*C0
#            Cstern = (params.epsilonStern/params.epsilonR)*(lambda_d/params.lambda_Stern)*C0
            
            V_T=(params.kB*params.T)/params.eCharge #Thermal Voltage
            #print(model.circuitModel)
            #Define parameters relevant to circuit model    
            circuitParams=params
            circuitParams.C0=C0
            circuitParams.Cox=Cox
            circuitParams.Cstern=Cstern
            #circuitParams.CB=CBulk
            circuitParams.R=R
            
            delta_phi0=Vpp/V_T #Make applied voltage dimensionless  
            #print(delta_phi0)
#            nFreq = len(omegaDimIn); 
#            class initConditions:
#                c0Init=np.reshape(np.array([c0*1.0]*nFreq),(nFreq,1));
#                R0Init=np.reshape(np.array([R*1.0]*nFreq),(nFreq,1));
#                q0Init=np.reshape(np.array([0.0]*nFreq),(nFreq,1)); #0;
#                v0Init=np.reshape(np.array([0.0]*nFreq),(nFreq,1)); #0;
#                vOxInit=np.reshape(np.array([0.0]*nFreq),(nFreq,1)); #0;
#                vSternInit=np.reshape(np.array([0.0]*nFreq),(nFreq,1)); #0;
#                vBInit=np.reshape(np.array([0.0]*nFreq),(nFreq,1));#0;#delta_phi0

            class initConditions:
                c0Init=np.array([c0*1.0]);
                R0Init=np.array([R*1.0]);
                q0Init=np.array([0.0]); #0;
                v0Init=np.array([0.0]); #0;
                vOxInit=np.array([0.0]); #0;
                vSternInit=np.array([0.0]); #0;
                vBInit=np.array([0.0]);#0;#delta_phi0
            displ = np.zeros((len(omegaDimIn),1),dtype=float)
            #frequency = omegaDimIn/(params.Di/(params.L**2)) 
            uniqueOmega = np.unique(omegaDimIn); 
            
            t0=0; 
            nPeaks=20;
            for chooseFrequency in range(len(uniqueOmega)):
                idx = np.where(omegaDimIn==uniqueOmega[chooseFrequency])[0];
                frequency = uniqueOmega[chooseFrequency]/(params.Di/(params.L**2)); 
                tEnd=(nPeaks*2*np.pi)/frequency
                dt0=0.01/frequency; dt=dt0/(2**0);
                dx0=1./50; dx=dx0/(2**0);
                nTime=(tEnd-t0)/dt
                
                time=np.linspace(0,tEnd,nTime)
                m=int(1./dx); 
                x=np.linspace(0,1,m);
                initConditions.cInit=(2*c0*np.ones((m,1)))
                
                [chargeOut, currentOut, vEdlOut, vOxOut, vSternOut, cOut, cTotOut, ROut, vBulkOut,timeBulk]=run.implementCircuitModel(initConditions,circuitParams,model,time,dx,eps,frequency,delta_phi0)  
                zero_crossings = np.where(np.diff(np.signbit(vBulkOut)))[0]
                zero_crossings[0]=zero_crossings[0]-1
                zero_crossings[-1]=zero_crossings[-1]+1
                vBulkRMS=np.sqrt(integrate.cumtrapz(vBulkOut[zero_crossings[-3]:zero_crossings[-1]]**2,time[zero_crossings[-3]:zero_crossings[-1]])/(time[zero_crossings[-3]+1:zero_crossings[-1]]-time[zero_crossings[-3]]))*V_T 
                #displ[chooseFrequency]= params.epsilonR*(alpha)*(vBulkRMS[-1]**2)   
                displ[idx]= (1.0/k)*((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g))*(vBulkRMS[-1]**2);  

            
            
            #print(tEnd,dt0)
#            nTime=(tEnd-t0)/dt
#            time=np.linspace(0,tEnd,nTime)
#            m=int(1./dx) 
#            x=np.linspace(0,1,m)
#            initConditions.cInit=(2*c0*np.ones((m,1)))
#            nFreq = len(frequency)
#            #print(frequency,omegaDimIn[chooseFrequency],lambda_d,eps,model.circuitModel)
#            #stateOut = run.implementCircuitModel(initConditions,circuitParams,model,time,dx,eps,frequency,delta_phi0);
#            [chargeOut, currentOut, vEdlOut, vOxOut, vSternOut, cOut, cTotOut, ROut, vBulkOut,timeBulk]=run.implementCircuitModel(initConditions,circuitParams,model,time,dx,eps,frequency,delta_phi0)  
#            #[chargeOut, currentOut, vEdlOut, vOxOut, vSternOut, cOut, cTotOut, ROut, vBulkOut] = [stateOut[:,(idx-1)*nFreq:idx*nFreq] for idx in range(1,9)]
#            #print("vBulk",vBulkOut,np.min(vBulkOut))
#            time=timeBulk;
#           # print("timeBulk",timeBulk)
#            for chooseFrequency in xrange(len(omegaDimIn)):
#                #plt.plot(timeBulk,vBulkOut[:,chooseFrequency])
#                zero_crossings = np.where(np.diff(np.signbit(vBulkOut[:,chooseFrequency])))[0]
#                zero_crossings[0]=zero_crossings[0]-1
#                zero_crossings[-1]=zero_crossings[-1]+1
#                vBulkRMS=np.sqrt(integrate.cumtrapz(vBulkOut[zero_crossings[-3]:zero_crossings[-1],chooseFrequency]**2,time[zero_crossings[-3]:zero_crossings[-1]])/(time[zero_crossings[-3]+1:zero_crossings[-1]]-time[zero_crossings[-3]]))*V_T 
#                #displ[chooseFrequency]= params.epsilonR*(alpha)*(vBulkRMS[-1]**2)   
#                displ[chooseFrequency]= (1.0/k)*((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g))*(vBulkRMS[-1]**2);   
                
#            for chooseFrequency in xrange(len(omegaDimIn)):
#                #chooseFrequency=5
#                frequency = omegaDimIn[chooseFrequency]/(params.Di/(params.L**2)) 
#                
#                t0=0
#                nPeaks=20
#                tEnd=(nPeaks*2*np.pi)/frequency
#                #dt0=0.0025/frequency
#                #dx0=1./200
#                dt0=0.01/frequency
#                dx0=1./50
#                
#                dt=dt0/(2**0)
#                dx=dx0/(2**0)
#                nTime=(tEnd-t0)/dt
#                time=np.linspace(0,tEnd,nTime)
#                m=int(1./dx) 
#                x=np.linspace(0,1,m)
#                initConditions.cInit=(2*c0*np.ones((m,1)))
#                #print(frequency,omegaDimIn[chooseFrequency],lambda_d,eps,model.circuitModel)
#                [chargeOut, currentOut, vEdlOut, vOxOut, vSternOut, cOut, cTotOut, ROut, vBulkOut,timeBulk]=run.implementCircuitModel(initConditions,circuitParams,model,time,dx,eps,frequency,delta_phi0)  
#                zero_crossings = np.where(np.diff(np.signbit(vBulkOut)))[0]
#                zero_crossings[0]=zero_crossings[0]-1
#                zero_crossings[-1]=zero_crossings[-1]+1
#                vBulkRMS=np.sqrt(integrate.cumtrapz(vBulkOut[zero_crossings[-3]:zero_crossings[-1]]**2,time[zero_crossings[-3]:zero_crossings[-1]])/(time[zero_crossings[-3]+1:zero_crossings[-1]]-time[zero_crossings[-3]]))*V_T 
#                #displ[chooseFrequency]= params.epsilonR*(alpha)*(vBulkRMS[-1]**2)   
#                displ[chooseFrequency]= (1.0/k)*((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g))*(vBulkRMS[-1]**2);    
##                plt.figure(); plt.plot(timeBulk*((params.L**2)/params.Di),vBulkOut) 
##                plt.figure(); plt.plot(timeBulk*((params.L**2)/params.Di),vEdlOut) 
##                plt.figure(); plt.plot(timeBulk*((params.L**2)/params.Di),ROut) 
            return displ.ravel()
        return variableResistorCircuitModelFit

    def variableResistorCircuitROxModel(self,Vpp,params,combDriveParams,model):
        def variableResistorCircuitROxModelFit(omegaDimIn,alphaOx,alphaStern,ROx,c0NomIn,k):
            k=k*1e-6
            run=runModels.implementationClass();
            NFingers = combDriveParams.NFingers;
            NCombs = combDriveParams.NCombs;
            epsilonBulk = params.epsilonR;
            epsilon0  = params.epsilon0;
            epsilonOx = params.epsilonOx;
            b = combDriveParams.b;
            g = combDriveParams.d;
            c0NomIn=c0NomIn*100e-6
            c0NomIn=(c0NomIn*1000.0)*params.NA
            lambda_d=np.sqrt((params.epsilonR*params.epsilon0*params.kB*params.T)/(2*(params.eCharge**2)*c0NomIn)) 
            c0NomIn=c0NomIn/(params.NA*1000)
            eps=lambda_d/params.L
            
            oxideLayer=1.0
            sternLayer=1.0
            doubleLayer=1.0
#            bulkCapacitance=params.bulkCapacitance
        
            #Define Resistance(s) and Capacitance(s)
            C0=2*eps*doubleLayer #Dimensionless linear component electric double layer capacitor
            c0=1 #Dimensionless bulk concentration of single ion
            R=1./(2*c0) #Dimensionless initial resistance of bulk       
            Cox=alphaOx*C0*oxideLayer  #Dimensionless capacitance of oxide
            Cstern=alphaStern*C0*sternLayer #Dimensionless capacitance of Stern Layer
            #CBulk=(params.epsilonR/params.epsilonR)*(lambda_d/params.L)*C0*bulkCapacitance
                    
            V_T=(params.kB*params.T)/params.eCharge #Thermal Voltage
                
            #Define parameters relevant to circuit model    
            circuitParams=params
            circuitParams.C0=C0
            circuitParams.Cox=Cox
            circuitParams.ROx=ROx
            circuitParams.Cstern=Cstern
            #circuitParams.CB=CBulk
            circuitParams.R=R
             
            delta_phi0=Vpp/V_T #Make applied voltage dimensionless  
#            class initConditions:
#                q0Init=0;
#                v0Init=0;
#                vOxInit=0;
#                vSternInit=0;
#                vBInit=0;#delta_phi0
            class initConditions:
                c0Init=np.array([c0*1.0]);
                R0Init=np.array([R*1.0]);
                q0Init=np.array([0.0]); #0;
                v0Init=np.array([0.0]); #0;
                vOxInit=np.array([0.0]); #0;
                vSternInit=np.array([0.0]); #0;
                vBInit=np.array([0.0]);#0;#delta_phi0
            displ = np.zeros((len(omegaDimIn),1),dtype=float)          
            uniqueOmega = np.unique(omegaDimIn); 
            
            t0=0; 
            nPeaks=20;
            for chooseFrequency in range(len(uniqueOmega)):
                idx = np.where(omegaDimIn==uniqueOmega[chooseFrequency])[0];
                frequency = uniqueOmega[chooseFrequency]/(params.Di/(params.L**2)); 
                tEnd=(nPeaks*2*np.pi)/frequency
                dt0=0.01/frequency; dt=dt0/(2**0);
                dx0=1./50; dx=dx0/(2**0);
                nTime=(tEnd-t0)/dt
                
                time=np.linspace(0,tEnd,nTime)
                m=int(1./dx); 
                x=np.linspace(0,1,m);
                initConditions.cInit=(2*c0*np.ones((m,1)))
                
                [chargeOut, currentOut, vEdlOut, vOxOut, vSternOut, cOut, cTotOut, ROut, vBulkOut,timeBulk]=run.implementCircuitModel(initConditions,circuitParams,model,time,dx,eps,frequency,delta_phi0)  
                zero_crossings = np.where(np.diff(np.signbit(vBulkOut)))[0]
                zero_crossings[0]=zero_crossings[0]-1
                zero_crossings[-1]=zero_crossings[-1]+1
                vBulkRMS=np.sqrt(integrate.cumtrapz(vBulkOut[zero_crossings[-3]:zero_crossings[-1]]**2,time[zero_crossings[-3]:zero_crossings[-1]])/(time[zero_crossings[-3]+1:zero_crossings[-1]]-time[zero_crossings[-3]]))*V_T 
                #displ[chooseFrequency]= params.epsilonR*(alpha)*(vBulkRMS[-1]**2)   
                displ[idx]= (1.0/k)*((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g))*(vBulkRMS[-1]**2); 
#                plt.plot(time,vBulkOut)
                
#            for chooseFrequency in xrange(len(omegaDimIn)):
#                #chooseFrequency=5
#                frequency = omegaDimIn[chooseFrequency]/(params.Di/(params.L**2)) 
#                
#                t0=0
#                nPeaks=20
#                tEnd=(nPeaks*2*np.pi)/frequency
#                #dt0=0.0025/frequency
#                #dx0=1./200
#                dt0=0.01/frequency
#                dx0=1./50
#                
#                dt=dt0/(2**0)
#                dx=dx0/(2**0)
#                nTime=(tEnd-t0)/dt
#                time=np.linspace(0,tEnd,nTime)
#                m=int(1./dx); 
#                x=np.linspace(0,1,m)
#                initConditions.cInit=(2*c0*np.ones((m,1)))
#                [chargeOut, currentOut, vEdlOut, vOxOut, vSternOut, cOut, cTotOut, ROut, vBulkOut,timeBulk]=run.implementCircuitModel(initConditions,circuitParams,model,time,dx,eps,frequency,delta_phi0)  
#                zero_crossings = np.where(np.diff(np.signbit(vBulkOut)))[0]
#                zero_crossings[0]=zero_crossings[0]-1
#                zero_crossings[-1]=zero_crossings[-1]+1
#                vBulkRMS=np.sqrt(integrate.cumtrapz(vBulkOut[zero_crossings[-3]:zero_crossings[-1]]**2,time[zero_crossings[-3]:zero_crossings[-1]])/(time[zero_crossings[-3]+1:zero_crossings[-1]]-time[zero_crossings[-3]]))*V_T 
#                #displ[chooseFrequency]= params.epsilonR*(alpha)*(vBulkRMS[-1]**2)   
#                displ[chooseFrequency]= (1.0/k)*((NFingers*NCombs*epsilon0*epsilonBulk*b)/(g))*(vBulkRMS[-1]**2);    
#                #plt.plot(timeBulk*((params.L**2)/params.Di),vBulkOut) 
            return displ.ravel()
        return variableResistorCircuitROxModelFit
   