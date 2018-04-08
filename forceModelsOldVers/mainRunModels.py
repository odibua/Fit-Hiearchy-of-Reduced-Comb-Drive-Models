#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:52:54 2018

@author: odibua
"""
import matplotlib.pyplot as plt
import matplotlib as mplib
import numpy as np
from numpy import inf
from scipy.optimize import curve_fit
from runReducedForceModels import *

modelsString = ['Classic','LeakyOx','LeakyOxStern','VariableRes','LeakyOxStern+VariableRes'];

runModels = runReducedModels();


c0NomIn0 = 10;  
k0=1.0e6;
Vpp = 2.6;

nFreq = 10;
freqDim = np.logspace(3,6,nFreq);
omegaDim = freqDim*2*np.pi;#np.array([freqDim[0]*2*np.pi]);
#omegaDim = np.array([freqDim[0]*2*np.pi]);



modelsUse = modelsString[0];
modelParams,modelFunc = runModels.reducedModels(modelsUse,c0NomIn0,k0,Vpp)
displClassic = modelFunc(omegaDim,*modelParams);

modelsUse = modelsString[2];
modelParams,modelFunc = runModels.reducedModels(modelsUse,c0NomIn0,k0,Vpp)
displLeakyOx= modelFunc(omegaDim,*modelParams);  


