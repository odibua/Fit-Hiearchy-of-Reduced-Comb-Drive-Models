#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:33:18 2018

@author: odibua
"""
import numpy as np
import reducedForceModels
from reducedForceModels import calcTimeConstantOxideOnly
#Introduce class that contains combdrive fit functions and circuit model functions


#Define parameters relevant to the combdrive actuator and to the electrolyte 
class runReducedModels():
    def __init__(self,t,g,NF,NC):
        class params:
            #Define universal constants and Temperature
            eCharge=1.602177e-19 #Charge of electron(C)
            NA = 6.02214e23 #Avogadro's Number(num/mol)
            kB=1.38065e-23 #Boltzmann Constant
            epsilon0=8.85419e-12 #Permitivitty of Free Space
            T=273+25 #Temperature (K)
                    
            #Define medium constants and thickness of mediums
            lambda_Ox=2e-9 #Thickness of Oxide Layers
            lambda_Stern=0.5e-9 #Thickness of Stern Layers
            L=g;#5e-6  #Distance between Electrodes
            #L=5e-6  #Distance between Electrodes
            epsilonR=78 #Relative Permitivitty of Water
            epsilonOx=3.5 #Relative Permitivitty of Oxide
            epsilonStern=5.0 #Relative Permitivitty of Stern Layer
            Di = (1.93e-5)*((1e-2)**2) #Estimate of diffusivity of KCl
        class combDriveParams:
            NFingers=NF;#25; #Number of comb fingers
            NCombs=NC;#4; #NEED TO CHECK THIS
            b=t;#15e-6; #Electrode Thickness
            d=g;#5e-6; #Comb gap
            
        class model:
            NormalCircuit=0
            ReducedCircuit=1
        
        self.params = params;
        self.combDriveParams = combDriveParams;
        self.model = model;
        self.combDriveModels=reducedForceModels.combDriveModels()
    
    def reducedModels(self,modelsUse,c0In0,k0,Vpp):
        combDriveModels = self.combDriveModels;
        params = self.params;
        combDriveParams = self.combDriveParams;
        VRMS = Vpp/np.sqrt(2.0);
        model = self.model;
        tauIn0,tauInNonDim,COxNonDim = calcTimeConstantOxideOnly(c0In0*100e-6,params)
        if (modelsUse == 'Classic'):
            epsilon0 = params.epsilon0;
            epsilonOx = params.epsilonOx
            lambdaOx = params.lambda_Ox; 
            COx0 = (epsilon0*epsilonOx)/(lambdaOx);
            RBulk0 = tauIn0/COx0;
            #k0=4.3;
            #tauIn0  = tauNomIn0; 
            #modelParams = (tauIn0,k0);
            modelParams = (COx0,RBulk0,params.L,k0);
            func = combDriveModels.classicCircuitModel(VRMS,combDriveParams,params);    
        elif (modelsUse == 'LeakyOx'):
            epsilon0 = params.epsilon0;
            epsilonOx = params.epsilonOx
            lambdaOx = params.lambda_Ox; 
            CBulk = (params.epsilon0*params.epsilonR)/params.L
            #k0 = 4.3;
            COx0 = (epsilon0*epsilonOx)/(lambdaOx);
            tauOx = tauIn0;
            ROx0 = tauOx/COx0;
            RBulk0 = 0.1*tauIn0/COx0;#0.8*ROx0; #tauBulk0 = tauOx/1000;
            #print(ROx0,RBulk0)
            #modelParams = (ROx0,COx0,RBulk0,k0);
            modelParams = (ROx0,COx0,RBulk0,params.L,k0);
            func = combDriveModels.leakyDielectricModel(VRMS,combDriveParams,params);       
        elif (modelsUse == 'LeakyOxStern'):
            epsilon0 = params.epsilon0;
            epsilonOx = params.epsilonOx;
            lambdaOx = params.lambda_Ox; 
            
            #k0 = 4.3;
            COx0 = (epsilon0*epsilonOx)/(lambdaOx);
            tauOx = tauIn0; 
            ROx0 = tauOx/COx0;
            
            RBulk0 = tauIn0/COx0; #tauBulk0 = tauOx/1000;
            CStern0 = (epsilon0*params.epsilonStern)/params.lambda_Stern;
            
            lambdad0=np.sqrt((params.epsilonR*params.epsilon0*params.kB*params.T)/(2*(params.eCharge**2)*c0In0*100e-6*1000.0*params.NA))
            #CEDL0 =1000*( epsilon0*params.epsilonR)/lambdad0;
            #print(CStern0,CEDL0)
            #modelParams = (ROx0,COx0,CStern0,RBulk0,k0);   
            modelParams = (ROx0,COx0,CStern0,RBulk0,params.L,k0);  
            func = combDriveModels.leakyDielectricSternModel(VRMS,combDriveParams,params);
        elif (modelsUse == 'VariableRes'): 
            LinCapModel=0
            NonLinCapModel=1
            NonLinCapVarResModel=2
            NonLinCapInhomResModel=3
            NonLinCapVarResROxModel=4
            
            model.LinCapModel=LinCapModel;
            model.NonLinCapModel=NonLinCapModel;
            model.NonLinCapVarResModel=NonLinCapVarResModel;
            model.NonLinCapInhomResModel=NonLinCapInhomResModel;
            model.NonLinCapVarResROxModel = NonLinCapVarResROxModel;
            
            model.circuitModel=model.NonLinCapVarResModel;
            model.simulationType=model.NormalCircuit;
            
            epsilonBulk = params.epsilonR;
            epsilon0  = params.epsilon0;
            epsilonOx = params.epsilonOx;
            #print("c0",c0In0)
            c0NomIn=c0In0*100e-6;
            #print(c0NomIn)
            c0NomIn=(c0NomIn*1000.0)*params.NA
            lambda_d=np.sqrt((params.epsilonR*params.epsilon0*params.kB*params.T)/(2*(params.eCharge**2)*c0NomIn)) 
            eps=lambda_d/params.L
         
            #Define Resistance(s) and Capacitance(s)
            C0=2*eps #Dimensionless linear component electric double layer capacitor     
            Cox=(params.epsilonOx/params.epsilonR)*(lambda_d/params.lambda_Ox)*C0 #Dimensionless capacitance of oxide
            Cstern=(params.epsilonStern/params.epsilonR)*(lambda_d/params.lambda_Stern)*C0 #Dimensionless capacitance of Stern Layer
            modelParams = (Cox,Cstern,c0In0,params.L,k0);
            func = combDriveModels.variableResistorCircuitModel(Vpp,params,combDriveParams,model);           
        elif (modelsUse == 'LeakyOxStern+VariableRes'):
            epsilon0 = params.epsilon0;
            epsilonOx = params.epsilonOx;
            lambdaOx = params.lambda_Ox; 
            c0NomIn=c0In0*100e-6
            c0NomIn=(c0NomIn*1000.0)*params.NA
            lambda_d=np.sqrt((params.epsilonR*params.epsilon0*params.kB*params.T)/(2*(params.eCharge**2)*c0NomIn)) 
            c0NomIn=c0NomIn/(params.NA*1000)
            eps=lambda_d/params.L
             
            
            LinCapModel=0 
            NonLinCapModel=1
            NonLinCapVarResModel=2
            NonLinCapInhomResModel=3
            NonLinCapVarResROxModel=4
                
            model.LinCapModel=LinCapModel
            model.NonLinCapModel=NonLinCapModel
            model.NonLinCapVarResModel=NonLinCapVarResModel
            model.NonLinCapInhomResModel=NonLinCapInhomResModel
            model.NonLinCapVarResROxModel = NonLinCapVarResROxModel;
            
            model.circuitModel=model.NonLinCapVarResROxModel
            model.simulationType=model.NormalCircuit
            
            C0=2*eps #Dimensionless linear component electric double layer capacitor     
            Cox=(params.epsilonOx/params.epsilonR)*(lambda_d/params.lambda_Ox)*C0 #Dimensionless capacitance of oxide
            Cstern=(params.epsilonStern/params.epsilonR)*(lambda_d/params.lambda_Stern)*C0 #Di
            tauOx = tauIn0; 
             
            ROx0 = tauInNonDim/COxNonDim
            alphaOx0 = 1;  
            alphaStern0 = 1; 
             
            modelParams = (Cox,Cstern,ROx0,c0In0,params.L,k0)
            func = combDriveModels.variableResistorCircuitROxModel(Vpp,params,combDriveParams,model);
        
        return(modelParams,func)