#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:51:56 2018

@author: odibua
"""
import sys
sys.path.append('./particleSwarmOptimizationMethods/')


import numpy as np
from numpy import inf
import matplotlib as mplib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from psoMethods import PSO
from psoParticles import psoParticle
from npsoParticles import npsoParticle 
from npsoInterpFuncs import npsoInterpFunc 
import multiprocessing

#from runReducedForceModels import *
#from fitModelsToData import fitModelsToData
from fitModelsToData import evaluateFitnessFunctions

def fitModels(nTrials,numIters,posMin,posMax,velMin,velMax,methodUse,weights,thresh,modelFunc,yData,errFunc):
    #Define PSO Parameters
    numParticles=25;
    neighborSize = 2#NPSO Parameter
    w=1.0;
    tol=1e-3;
    #numIters=2000
    numEvalState=2;
    kappa = 0.5;
    mult=1;
    c1=2.0
    c2 = c1*mult;
    constrict=1.0
    optimType='Min';
    
    output = [];#[[0.0]*nTrials];
    fitness = [[0.0]*nTrials];
    parameters = [[0.0]*nTrials]; 
    #Call PSO class
    pso=PSO();
    
    for k in range(nTrials):
        print("Trial",k+1)
        #Execute PSO
        if (methodUse == 'PSO'):
            output.append(pso.executePSO(c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,errFunc,evaluateFitnessFunctions))
        elif (methodUse == 'GCPSO'):
            c1=2.05; c2=c1;
            output.append(pso.executeGCPSO(constrict,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,errFunc,evaluateFitnessFunctions))
        elif (methodUse == 'NPSO'):
            output.append(pso.executeNPSO(neighborSize,w,posMin,posMax,velMin,velMax,numIters,numParticles,npsoParticle,optimType,numEvalState,errFunc,evaluateFitnessFunctions,npsoInterpFunc))

    x=output[0][1][1][0];
    parameters = np.array([output[k][1][1][1:] for k in range(nTrials)]);
    fitness = np.array([output[k][1][0] for k in range(nTrials)]);
    if (optimType.lower()=='min'):
        rankedPSOParams  = parameters[np.argsort(fitness)];
    else: 
        rankedPSOParams  = parameters[np.argsort(fitness)];
    
    rankedFitness = fitness[np.argsort(fitness)];
    

    sigma = np.ones((np.shape(yData.ravel())));
    sigma[yData.ravel()<thresh] = 1.0/weights;#np.sqrt(10)
    #print("rankedPSOParams",rankedPSOParams,"realParams",10**rankedPSOParams)
    #print("x",parametersRanked[0][0],"params",parametersRanked[0][1:])
    #print("Data",np.array(yData.ravel()),"params",parametersRanked[0][1:],"freq",parametersRanked[0][0])

    #finalParams, pcov = curve_fit(modelFunc, x,  np.array(yData.ravel()),p0=tuple(rankedPSOParams[0]),sigma=sigma,absolute_sigma=False )
#    finalParams, pcov = curve_fit(modelFunc, x,  np.array(yData.ravel()),p0=tuple(rankedPSOParams[0]),bounds=(0, [float("inf")]*len(posMax[1:])),sigma=sigma,absolute_sigma=False )
#    print("posMin",posMin.tolist())
    rankedPSOGradientParams = [];
    for j in range(len(rankedPSOParams)):
        paramsTemp,_ = curve_fit(modelFunc, x,  np.array(yData.ravel()),p0=tuple(rankedPSOParams[0]),bounds=(posMin[1:].tolist(),posMax[1:].tolist()),sigma=sigma,absolute_sigma=False )
        rankedPSOGradientParams.append(paramsTemp);
        
    finalParams, pcov = curve_fit(modelFunc, x,  np.array(yData.ravel()),p0=tuple(rankedPSOParams[0]),bounds=(posMin[1:].tolist(),posMax[1:].tolist()),sigma=sigma,absolute_sigma=False )
    
             #   tau500uMOpt, pcov = curve_fit(combDriveDispl.makeVikramModelUseAirAlpha(alphaOpt,VRMS), omegaDimIn500uMKClStack.ravel(),  KCL500uMDat.ravel(),p0=tauIn0500uMKCl,sigma=sigma,absolute_sigma=False )
            #funcKCLDispl=combDriveDispl.makeVikramModelUseAirAlpha(alphaOpt,VRMS)
    print("rankedPSOParams",rankedPSOParams,"Final params",finalParams)
    
    return (finalParams,rankedPSOParams,rankedFitness,rankedPSOGradientParams)
