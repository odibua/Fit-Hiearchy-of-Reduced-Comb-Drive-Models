"""
Created on Wed Jan 31 21:08:41 2018
 
@author: odibua
"""
import numpy as np
from callFitModels import callFitModel
 
#Device and concentration string arrays
concentrationString = ['100','1000','500','5000','10000'];#['10000','1000','100','500','5000',];
#concentrationString = ['500','5000'];
#Type of model fitting
fitModelString = ['L2'];#,'L2'];
 
#Models to choose from
modelString = ['Classic','LeakyOx','LeakyOxStern'];#,'VariableRes','LeakyOxStern+VariableRes'];
#modelString = ['VariableRes','LeakyOxStern+VariableRes'];
 
#PSO method to choose from
methodString = ['GCPSO'];
 
Vpp = 2;
dataFileName = str(int(2*Vpp))+'Vpp_Device12.dat';
   
gArr = np.array([5.0e-6,2e-6]); 
bArr = np.array([15e-6,15e-6]); 
NFingersArr = np.array([25,25]); 
NCombsArr = np.array([4,8]);
deviceString = ['deviceT3','deviceT4']; 
#deviceString = ['deviceT3'];#,'deviceT4'];
   
b={}; g={}; NFingers={}; NCombs={};
for l in range(len(deviceString)):
    b[deviceString[l]] = bArr[l];
    g[deviceString[l]] = gArr[l];
    NFingers[deviceString[l]] = NFingersArr[l];
    NCombs[deviceString[l]] = NCombsArr[l];
     
nTrials=10;
numIters=1000;
callFitModel(nTrials,numIters,Vpp,b,g,NFingers,NCombs,deviceString,concentrationString,fitModelString,modelString,methodString,dataFileName)