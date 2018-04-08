#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:47:41 2018

@author: odibua
"""    
import sys
import copy
import timeit
sys.path.append('./forceModels/')

from numpy import inf
import numpy as np

#import matplotlib as mplib
import matplotlib.pyplot as plt

from runReducedForceModels import *
from fitModelsToData import modelToDataErr
from fitReducedModels import fitModels  

#dataFileName = '4Vpp_Device12.dat';
def callFitModel(nTrials,numIters,Vpp,b,g,NFingers,NCombs,deviceString,concentrationString,fitModelString,modelString,methodString,dataFileName):    
    for fitModelType in fitModelString:        
        #Choose type of PSO and set parameters
        for methodUse in methodString:
            thresh = 0.1;
            weights = 1e-10;
            
            #Select data to load and initialize relevant parameters
            #device = deviceString[0];
            for device in deviceString:
                #Initialize run models class
                runModels = runReducedModels(b[device],g[device],NFingers[device],NCombs[device]);   
                #concentration = concentrationString[0];
                for concentration in concentrationString:
                    c0NomIn0 = float(concentration)/100; 
                    #Vpp = 2.0;  
                    k0 = 1;
                      
                    #Load data
                    data = np.loadtxt('./data/'+device+'/'+concentration+'uMKCl'+'/'+dataFileName); 
                    displData = data[1:,0:]; 
                    freqDim = data[0,0:];
                    freqDimStack = np.tile(freqDim,displData.shape[0]);
                    omegaDimStack = freqDimStack*2*np.pi;
                    omegaDimStack = omegaDimStack.ravel();
                     
                    #Select the model, load the model parameters, and the model function
                    #modelUse = modelsString[2];
                    for modelUse in modelString:
                        modelParams,modelFunc = runModels.reducedModels(modelUse,c0NomIn0,k0,Vpp); 
                        modelParams = list(modelParams); modelParams.insert(0,omegaDimStack);
                        modelParams = tuple(modelParams);
                        #out = modelFunc(*modelParams)
                        
                        errFunc = modelToDataErr(displData,modelFunc,weights,thresh);
                         
                        if (fitModelType == 'L2'):
                            errFunc = errFunc.L2Norm; 
                        elif (fitModelType == 'L1'):
                            errFunc = errFunc.L1Norm; 
                         
                        posMax = copy.deepcopy(np.array(modelParams));
                        posMin = copy.deepcopy(np.array(modelParams));
                        velMax = copy.deepcopy(posMax); 
                        velMin = copy.deepcopy(posMin); 
                        #print(posMin);
                            
                        if (modelUse == 'Classic' or modelUse == 'LeakyOx' or modelUse == 'LeakyOxStern'):
                            posMax[1:] = np.array([modParams*15 for modParams in list(modelParams[1:])]);
                            posMin[1:] = np.array([modParams*1e-2 for modParams in list(modelParams[1:])]);
                            posMax[1:] = np.log10(posMax[1:].astype(np.float64)); posMin[1:] = np.log10(posMin[1:].astype(np.float64))
                        elif (modelUse == 'VariableRes'):
#                            modelParams = np.array(modelParams); 
                            if (int(concentration) == 100):
                                boundsUp = 3; boundsLow=0.01;
                            else:
                                boundsUp = 5; boundsLow=0.01;   
                            posMaxTmp = copy.deepcopy(posMax[1:]); posMinTmp = copy.deepcopy(posMin[1:]);
                            boundsConcUp=1.5; boundsConcDown=0.5; boundsKUp=10; boundsKDown=0.01; 
                            boundsUpL = 2.2; boundsLowL=1;
                            
                            idxCap = np.array([0,1]); idxL=3; idxConc=2; idxK=4;
                            posMaxTmp[idxCap]=posMaxTmp[idxCap]*boundsUp; posMinTmp[idxCap]=posMinTmp[idxCap]*boundsLow;
                            posMaxTmp[idxL]=posMaxTmp[idxL]*boundsUpL; posMinTmp[idxL]=posMinTmp[idxL]*boundsLowL;                            
                            posMaxTmp[idxConc] = posMaxTmp[idxConc]*boundsConcUp; posMinTmp[idxConc] = posMinTmp[idxConc]*boundsConcDown;
                            posMaxTmp[idxK] = posMaxTmp[idxK]*boundsKUp; posMinTmp[idxK] = posMinTmp[idxK]*boundsKDown;
                            
                            posMinTmp = posMinTmp.astype(float); posMaxTmp = posMaxTmp.astype(float);
                            posMinTmp = np.log10(posMinTmp.astype(np.float64)); posMaxTmp = np.log10(posMaxTmp.astype(np.float64));
                            posMin[1:]=posMinTmp; posMax[1:]=posMaxTmp;
                        elif (modelUse == 'LeakyOxStern+VariableRes'):
                            modelParams = np.array(modelParams); 
                            if (int(concentration) == 100):
                                boundsUp = 3; boundsLow=0.01;                                
                            else:
                                boundsUp = 5; boundsLow=0.01;
                            boundsConcUp=1.5; boundsConcDown=0.5; boundsUpROx=14; boundsLowROx=0.001; 
                            boundsUpL = 2.2;  boundsLowL=1.0; boundsKUp=10; boundsKDown=0.01; 
                            posMaxTmp = copy.deepcopy(posMax[1:]); posMinTmp = copy.deepcopy(posMin[1:]);
                            
                            idxCap = np.array([0,1]); idxROx=2; idxConc=3; idxL=4; idxK=5;
                            posMaxTmp[idxCap]=posMaxTmp[idxCap]*boundsUp; posMinTmp[idxCap]=posMinTmp[idxCap]*boundsLow;
                            posMaxTmp[idxROx]=posMaxTmp[idxROx]*boundsUpROx; posMinTmp[idxROx]=posMinTmp[idxROx]*boundsLowROx;
                            posMaxTmp[idxK] = posMaxTmp[idxK]*boundsKUp; posMinTmp[idxK] = posMinTmp[idxK]*boundsKDown;
                            posMaxTmp[idxL]=posMaxTmp[idxL]*boundsUpL; posMinTmp[idxL]=posMinTmp[idxL]*boundsLowL;
                            posMaxTmp[idxConc] = posMaxTmp[idxConc]*boundsConcUp; posMinTmp[idxConc] = posMinTmp[idxConc]*boundsConcDown;
                            
                            posMinTmp = posMinTmp.astype(float); posMaxTmp = posMaxTmp.astype(float);
                            posMinTmp = np.log10(posMinTmp.astype(np.float64)); posMaxTmp = np.log10(posMaxTmp.astype(np.float64));
                            posMin[1:]=posMinTmp; posMax[1:]=posMaxTmp   
                        
                        posMaxTmp = copy.deepcopy(posMax); posMinTmp = copy.deepcopy(posMin); 
                        velMax[1:] = [pos/10.0 for pos in posMax[1:]];
                        velMin[1:] = [pos/10.0 for pos in posMin[1:]]; 
    
                        t1 = timeit.default_timer();
                        finalParams,rankedPSOParams,rankedFitness,rankedPSOGradientParams = fitModels(nTrials,numIters,posMin,posMax,velMin,velMax,methodUse,weights,thresh,modelFunc,displData,errFunc);
                        finalParams = list(finalParams);
                        rankedPSOParams = list(rankedPSOParams);
                        rankedFitness = list(rankedFitness);
                        t2 = timeit.default_timer();
##                        
                        finalParams=list(finalParams);
#                        print("\n")
                        print("rankedFitness",rankedFitness,"rankedParams",rankedPSOParams,"finalParams",finalParams)
                        #plt.figure()
                        #finalParams.insert(0,omegaDimStack)
                        #plt.semilogx(omegaDimStack/(2*np.pi),displData.ravel(),'x',omegaDimStack/(2*np.pi),modelFunc(*finalParams),'s')
                        np.savez('./data/'+device+'/'+concentration+'uMKCl/'+fitModelType+'_'+methodUse+'_'+modelUse+'.npz',omegaDimStack=omegaDimStack,displData=displData,posMin=posMin,posMax=posMax,velMin=velMin,velMax=velMax,finalParams=finalParams,rankedPSOParams=rankedPSOParams,rankedFitness=rankedFitness,rankedPSOGradientParams=rankedPSOGradientParams)
#                    
##                    
#                
                
                
                
