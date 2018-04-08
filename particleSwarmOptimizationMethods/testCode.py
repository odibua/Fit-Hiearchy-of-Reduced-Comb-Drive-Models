#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:12:28 2017

@author: Ohi Dibua
Tests three different type of particle swarm optimization (PSO), written based on [1],
on a variety of test functions. The different types of particle swarm optimization are
standard PSO, GCPSO (constricted PSO that does not require bounds in velocity), and NPSO
(a non-parametric PSO that does not require the setting of coefficients). The last two methods
are generally more effective than the first.

[1] Behesti et al., "Non-parametric particle swarm optimization for global optimization",Applied Soft. Computing. 2015
"""
import sys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


from psoMethods import PSO
from psoParticles import psoParticle
from npsoParticles import npsoParticle 
from npsoInterpFuncs import npsoInterpFunc 

from testFuncs import rastriginFunc
from testFuncs import sphereFunc
from testFuncs import mcCormicFunc
from testFuncs import bealeFunc
from testFuncs import holderTableFunc
from testFuncs import rosenbrockFunc
from testFuncs import ackleysFunc
from testFuncs import evaluateFitnessFunctions

testString = ['sphere','ackleys','mccormic','beales','rastrigin'] 
methodString = ['PSO','GCPSO','NPSO'];
optimTypeString = ['max','min']
testUse = sys.argv[1]; methodUse = sys.argv[2]; optimType = sys.argv[3]; nDim = int(sys.argv[4]); numIters = int(sys.argv[5])

if (testUse not in testString):
    sys.exit('Not a valid test function. Input sphere, ackleys, mccormic, beales, or rastrigin'); 
elif (methodUse not in methodString):
    sys.exit('Not a valid PSO method. Input PSO, GCPSO, or NPSO.');
elif (optimType.lower() not in optimTypeString):
    sys.exit('Not a valid type of optimization. Input max or min');
elif (nDim <2):
    sys.exit('Dimension of inputs (which effects sphere, ackleys, and rastrigin) must be at least 2');
elif (numIters<10):
    sys.exit('Too little number of iterations in particle swarm optimization. Make at least 10 and increase iterations for more accuracy')
    
#testUse=testString[3]
#nDim=10
if (testUse=='sphere'):
##Sphere function parameters
    posMin = np.array([-10]*nDim).astype(float);
    posMax = np.array([10]*nDim).astype(float);
    velMin = posMin;
    velMax = posMax;
    funcUsed = sphereFunc
    trueOptimalFitness = 0;
    trueOptimalInput = np.array([0]*nDim)
elif (testUse=='ackleys'):
####Ackleys Function parameter inputs
    posMin = np.array([-32.768]*nDim);
    posMax = np.array([32.768]*nDim);
    velMin = posMin;
    velMax = posMax;
    funcUsed = ackleysFunc;
    trueOptimalFitness = 0;
    trueOptimalInput = np.array([0]*nDim)
elif (testUse=='mccormic'):
###McCormic Function parameter inputs
    posMin = np.array([-1.5,-3]);
    posMax = np.array([4.0,4.0]);
    posMin = [-1.5,-3];
    posMax = [4.0,4.0];
    velMin = posMin;
    velMax = posMax;
    funcUsed = mcCormicFunc;
    trueOptimalFitness = -1.9133;
    trueOptimalInput = np.array([-0.54719,-1.54719]);
elif (testUse=='beales'):
###Beale's Function parameter inputs
    posMin = np.array([-4.5,-4.5]);
    posMax = np.array([4.5,4.5]);
    posMin = [-4.5,-4.5];
    posMax = [4.5,4.5];
    velMin = posMin;#np.array([-1,-1]);
    velMax = posMax;#np.array([1,1]);
    funcUsed = bealeFunc;
    trueOptimalFitness = 0;
    trueOptimalInput = np.array([3,0.5])
elif (testUse=='rastrigin'):
#####Rastrigin Function parameter inputs
    posMin = np.array([-5.12]*nDim);
    posMax = anp.array([5.12]*nDim);
    #posMin = [-5.12,-5.12,-5.12];
    #posMax = [5.12,5.12,5.12];
    velMin = posMin;
    velMax = posMax;
    funcUsed = rastriginFunc
    trueOptimalFitness = 0;
    trueOptimalInput = np.array([0]*nDim)

#Define PSO Parameters
numParticles=40;
neighborSize = 2#NPSO Parameter
w=1.0;
tol=1e-3;
#numIters=1000#nFeatures*15;
numEvalState=2;
kappa = 0.5;
mult=1;
c1=2.0
c2 = c1*mult;
constrict=1.0
optimType='Min';

#Call PSO class
pso=PSO();

#Execute PSO
if (methodUse == 'PSO'):
    output=pso.executePSO(c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,funcUsed,evaluateFitnessFunctions)
elif (methodUse == 'GCPSO'):
    c1=2.05; c2=c1;
    output=pso.executeGCPSO(constrict,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,funcUsed,evaluateFitnessFunctions)
elif (methodUse == 'NPSO'):
    output=pso.executeNPSO(neighborSize,w,posMin,posMax,velMin,velMax,numIters,numParticles,npsoParticle,optimType,numEvalState,funcUsed,evaluateFitnessFunctions,npsoInterpFunc)

print('True Fitness',trueOptimalFitness,'Estimated Fitness',output[1][0])
print("\n")
print('True Optimal Input',  trueOptimalInput,'Optimal Input',output[1][1])