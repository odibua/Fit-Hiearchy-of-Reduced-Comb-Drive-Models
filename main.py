#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:47:41 2018

@author: odibua
"""    
import sys
sys.path.append('./forceModels/')

import numpy as np
from numpy import inf
import matplotlib as mplib
import matplotlib.pyplot as plt

from runReducedForceModels import *
from fitModelsToData import modelToDataErr
from fitReducedModels import fitModels  
import timeit





#Device and concentration string arrays
deviceString = ['deviceT3']#,'deviceT4'];
concentrationString = ['100']#,'500','1000','5000','10000'];
#concentrationString = ['100uMKCl','500uMKCl','1000uMKCl','5000uMKCl','10000uMKCl'];




dataFileName = '4Vpp_Device12.dat';
#def callFitModel(deviceString,concentrationString,dataFileName):
#Initialize run models class
runModels = runReducedModels();

#Type of model fitting
fitModelArr = ['L1']#,'L2'];

#Models to choose from
modelString = ['VariableRes'];#['Classic','LeakyOx','LeakyOxStern','VariableRes','LeakyOxStern+VariableRes'];

#PSO method to choose from
methodString = ['PSO']#,'GCPSO','NPSO'];

for fitModelType in fitModelArr:
    #Type of model fitting
    #fitModelType = fitModelArr[1];
    
    #Choose type of PSO and set parameters
    #methodUse = methodString[1];
    for methodUse in methodString:
        nTrials = 2;
        numIters = 10;
        thresh = 0.1;
        weights = 1e-10;
        
        #Select data to load and initialize relevant parameters
        #device = deviceString[0];
        for device in deviceString:
            #concentration = concentrationString[0];
            for concentration in concentrationString:
                c0NomIn0 = float(concentration)/100; 
                Vpp = 2.0;  k0 = 1;
                  
                #Load data
                data = np.loadtxt('./data/'+device+'/'+concentration+'uMKCl'+'/'+dataFileName); 
                #displData = data[1:]; 
                #freqDim = data[0]; 
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
                    start = timeit.timeit()
                    outTest = modelFunc(*modelParams);
                    end = timeit.timeit()
                    print("start",start,"end",end,"time",end-start)
#                        if (fitModelType == 'L2'):
#                            errFunc = errFunc.L2Norm; 
#                        elif (fitModelType == 'L1'):
#                            errFunc = errFunc.L1Norm; 
#                         
#                        posMax = np.array(modelParams);
#                        posMin = np.array(modelParams);
#                        velMax = posMax; 
#                        velMin = posMin; 
#                            
#                        if (modelUse == 'Classic' or modelUse == 'LeakyOx' or modelUse == 'LeakyOxStern'):
#                            posMax[1:] = np.array([modParams*1e2 for modParams in list(modelParams[1:])]);
#                            posMin[1:] = np.array([modParams*1e-6 for modParams in list(modelParams[1:])]);
#                        elif (modelUse == 'VariableRes'):
#                            posMax[1:3] = posMax[1:3]*1e1; posMax[-1] = posMax[-1]*1e1; posMax[3] = 100;
#                            posMin[1:3] = posMin[1:3]*1e-6; posMin[-1] = posMin[-1]*1e-6; posMin[3] = 0.5; 
#                        elif (modelUse == 'LeakyOxStern+VariableRes'):
#                            posMax[1:4] = posMax[1:4]*1e1; posMax[-1] = posMax[-1]*1e1; posMax[4] = 100;
#                            posMin[1:4] = posMin[1:4]*1e-6; posMin[-1] = posMin[-1]*1e-6; posMin[4] = 0.5;    
#                         
#                        velMax[1:] = [pos/10.0 for pos in posMax[1:]];
#                        velMin[1:] = [pos/10.0 for pos in posMin[1:]]; 
#                        
#                        #out = fitModels(nTrials,numIters,posMin,posMax,velMin,velMax,methodUse,weights,thresh,modelFunc,displData,errFunc)
#                         
#                        finalParams,rankedPSOParams,rankedFitness = fitModels(nTrials,numIters,posMin,posMax,velMin,velMax,methodUse,weights,thresh,modelFunc,displData,errFunc);
#                        finalParams = list(finalParams);
#                        rankedPSOParams = list(rankedPSOParams);
#                        rankedFitness = list(rankedFitness);
#                        
#                        np.savez('./data/'+device+'/'+concentration+'uMKCl/'+fitModelType+'_'+methodUse+'_'+modelUse+'.npz',omegaDimStack,displData,posMin,posMax,velMin,velMax,finalParams=finalParams,rankedPSOParams=rankedPSOParams,rankedFitness=rankedFitness)
#                    
#                    
            
                
                
                
