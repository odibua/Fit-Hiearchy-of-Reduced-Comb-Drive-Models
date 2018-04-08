#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:18:31 2018

@author: odibua
"""

import numpy as np

class modelToDataErr():
    def __init__(self,yData,modelFunc,weight,thresh):
        meanDat = np.mean(yData,axis=0);
        weightArr = np.ones(np.shape(meanDat))*1.0;       
        weightArr[meanDat<thresh] = weight;
        shapeYData = np.shape(yData);
        self.nData = shapeYData[0];
        self.n = shapeYData[0]*shapeYData[1];
        
        self.weightArr = np.tile(weightArr,shapeYData[0]);
        self.yData = yData;
        self.modelFunc = modelFunc;
        
    def L2Norm(self,modelParams):
        res = (self.yData.ravel()-self.modelFunc(*modelParams))/self.yData.ravel();
        return  np.sum((res**2)*self.weightArr.ravel())/self.n #((res**2)*self.weightArr)/self.n
     
    def L1Norm(self,modelParams):
        res = (self.yData.ravel()-self.modelFunc(*modelParams))#/self.yData.ravel();
        return np.sum(np.abs(res)*self.weightArr.ravel())/self.n 
    
    def L2NormFreq(self,modelParams):
        res = (self.yData-self.modelFunc(*modelParams))#/np.mean(self.yData,axis=0)#self.yData;#np.mean(self.yData,axis=0)
        resSign = -np.sign(np.sum(res,axis=0));
        #print(self.nData,np.sum(res/self.nData,axis=0),np.max(np.isnan(self.nData,np.sum(res/self.nData,axis=0))))
        return resSign*np.sqrt(np.sum((res**2)/self.nData,axis=0))#*100;
        #return np.sqrt(np.sum((res**2)/self.nData,axis=0)); 
        #return (np.sum(res/self.nData,axis=0));
    
    def L1NormFreq(self,modelParams):
        res = (self.yData-self.modelFunc(*modelParams))#/self.yData;
        return np.sqrt(np.sum(np.abs(res)/self.nData,axis=0))
        
    
def evaluateFitnessFunctions(optimType,currentState,localBestState,globalBestState=None):
    if (globalBestState==None):
        currentFitness=currentState[0];
        currentPos=currentState[1];
        
        localBestFitness=localBestState[0];
        localBestPos=localBestState[1];
        
        newLocalBool=0;
        if (optimType.lower()=='max'):
            if (currentFitness>localBestFitness):
                newLocalBool=1;
        elif (optimType.lower()=='min'):
            if (currentFitness<localBestFitness):
                newLocalBool=1;
        if (newLocalBool==1):
            localBestState=(currentFitness,currentPos)
        
        return localBestState
    
    elif (globalBestState is not None):
        localBestFitness=localBestState[0];
        localBestPos=localBestState[1];
        
        globalBestFitness=globalBestState[0];
        globalBestPos=globalBestState[1];
        
        newGlobalBool=0;
        if (optimType.lower()=='max'):
            if (localBestFitness>globalBestFitness):
                newGlobalBool=1;
        elif (optimType.lower()=='min'):
            if (localBestFitness<globalBestFitness):
                newGlobalBool=1;
        if (newGlobalBool==1):
#            print("New global",localBestFitness,localBestPos)
            globalBestState = (localBestFitness,localBestPos)
        
        return globalBestState        
#class fitModelsToData():
    