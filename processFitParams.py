#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:20:41 2018

@author: odibua
"""

import sys
import copy
import timeit
import collections
import matplotlib
import matplotlib.cm as cm

from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import FormatStrFormatter

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from matplotlib.patches import ConnectionPatch
from scipy import interpolate

sys.path.append('./forceModels/')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data

from runReducedForceModels import *
from fitModelsToData import modelToDataErr
from fitReducedModels import fitModels  
    
    
def errColorMap(b,g,NFingers,NCombs,fitModelString,methodString,deviceString,concentrationString,modelString):
    maxV = {'deviceT3':0.2,'deviceT4':0.2}
    minV = {'deviceT3':-0.2*0,'deviceT4':-0.2*0}
    colMap = cm.rainbow; #cm.gray
    
    for fitModelType in fitModelString:
        for methodUse in methodString:
            thresh = 0;
            weights = 1e-10;
            
            #Select data to load and initialize relevant parameters
            #device = deviceString[0];
            for device in deviceString:
                runModels = runReducedModels(b[device],g[device],NFingers[device],NCombs[device]);                  
                for concentration in concentrationString:
                    err = []
    #                plt.figure()
                    c0NomIn0 = float(concentration)/100; 
                    Vpp = 2.0;  k0 = 1;
                          
                    #Load data
                    data = np.loadtxt('./data/'+device+'/'+concentration+'uMKCl'+'/'+dataFileName); 
                    displData = data[1:,0:]; 
                    mnDispl = np.mean(displData,axis=0)
                    
                    idxUseData = np.where(mnDispl>thresh)[0];
                    mnDispl = mnDispl[idxUseData];
                    displData = data[1:,idxUseData]
                    freqDim = data[0,idxUseData];
                    freqDimLog = np.log10(freqDim)
                    omegaDim = 2*np.pi*freqDim;
                    x=freqDim;
                    print(x)
                    xMin = min(x); xMax = max(x);
                    
#                    xMin = min(mnDispl); xMax = max(mnDispl);
                    #yMin = 0; yMax = 0.5;
                    yMin = 0; yMax = 0.5;
                    for modelUse in modelString:
                        modelParams,modelFunc = runModels.reducedModels(modelUse,c0NomIn0,k0,Vpp); 
                        
                        errFunc = modelToDataErr(displData,modelFunc,weights,thresh);
                             
                        if (fitModelType == 'L2'):
                            errFunc = errFunc.L2NormFreq; 
                        elif (fitModelType == 'L1'):
                            errFunc = errFunc.L1NormFreq; 
                
                        out = np.load('./data/'+device+'/'+concentration+'uMKCl/'+fitModelType+'_'+methodUse+'_'+modelUse+'.npz');
                        finalParams = out["finalParams"]
                        if modelUse == 'LeakyOxStern':
                            #ROx0,COx0,CStern0,RBulk0,k0
                            ROx = 10**finalParams[0]; 
                            COx = 10**finalParams[1];                           
                            CStern = 10**finalParams[2];
                            RBulk = 10**finalParams[3];
                            modRange = np.array([0,0.5]);
                            freqAct = (1.0/(RBulk*COx))/(2*np.pi);
                            freqActMax = freqAct*10;
                            freqDiff = np.abs(freqDim-freqAct);
                            freqDiffMax = np.abs(freqDim-freqActMax);
                            idxFreqAct = np.where(freqDiff == np.min(freqDiff))[0];
                            print('IDXFreq',idxFreqAct)
                            idxFreqActMax = np.where(freqDiffMax == np.min(freqDiffMax))[0];
                            print(device,concentration,RBulk*COx,(1.0/(RBulk*COx))/(2*np.pi),RBulk/ROx)
                        finalParams = list(finalParams); finalParams.insert(0,omegaDim);
                        finalParams = tuple(finalParams)
                        err.append(errFunc(finalParams));
#                        plt.figure();
#                        plt.semilogx(freqDim,np.array(displData).T,freqDim,modelFunc(*finalParams),'*')
#                        print(err,np.shape(displData))
#                        #plt.legend(modelString)
#                        plt.title(device+concentration)
                        #err.append(np.sum(errFunc(finalParams),axis=0));
#                        err.append(np.sum(errFunc(finalParams),axis=0));
                    plt.figure();
                    plt.semilogx(freqDim,np.array(err).T)
#                    plt.semilogy(freqDimLog,np.array(err).T)
#                    print(err[1],np.shape(displData))
                    plt.legend(modelString)
                    plt.title(device+concentration)
#               
                    sepLines = np.array([[k]*len(freqDim) for k in range(3,len(modelString)*3,3)]) 
                    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(len(modelString), 1, sharex='col',sharey='row')
                    ax0.set_xscale('log');
                    ax1.set_xscale('log');
                    ax2.set_xscale('log');
                    ax3.set_xscale('log');
                    ax4.set_xscale('log');
                     
#                    im0 = ax0.imshow(np.reshape(err[0],(len(err[0]),1)).T, aspect='auto',vmin=0,vmax=maxV[device],extent=[xMin, xMax, yMin, yMax],   interpolation='bilinear',cmap=colMap )
#                    im1 = ax1.imshow(np.reshape(err[1],(len(err[1]),1)).T, aspect='auto',vmin=0,vmax=maxV[device],extent=[xMin, xMax, yMin, yMax],   interpolation='bilinear',cmap=colMap )
#                    im2 = ax2.imshow(np.reshape(err[2],(len(err[2]),1)).T, aspect='auto',vmin=0,vmax=maxV[device],extent=[xMin, xMax, yMin, yMax],   interpolation='bilinear',cmap=colMap )
#                    im3 = ax3.imshow(np.reshape(err[3],(len(err[3]),1)).T, aspect='auto',vmin=0,vmax=maxV[device], extent=[xMin, xMax, yMin, yMax],  interpolation='bilinear',cmap=colMap )
#                    im4 = ax4.imshow(np.reshape(err[4],(len(err[4]),1)).T, aspect='auto',vmin=0,vmax=maxV[device], extent=[xMin, xMax, yMin, yMax], interpolation='bilinear',cmap=colMap ) 
                    lnIDXUse = len(err[0]);
                    yPlaceHolder = np.linspace(0,0.5,len(freqDim));
                    im0 = ax0.pcolormesh(freqDim,yPlaceHolder,np.repeat(np.reshape(err[0],(len(err[0]),1)).T,lnIDXUse ,axis=0),shading='gouraud',vmin=minV[device],vmax=maxV[device], cmap=colMap )
                    im1 = ax1.pcolormesh(freqDim,yPlaceHolder,np.repeat(np.reshape(err[1],(len(err[1]),1)).T,lnIDXUse ,axis=0),shading='gouraud',vmin=minV[device],vmax=maxV[device], cmap=colMap )
                    im2 = ax2.pcolormesh(freqDim,yPlaceHolder,np.repeat(np.reshape(err[2],(len(err[2]),1)).T,lnIDXUse ,axis=0),shading='gouraud', vmin=minV[device],vmax=maxV[device],cmap=colMap )
                    im3 = ax3.pcolormesh(freqDim,yPlaceHolder,np.repeat(np.reshape(err[3],(len(err[3]),1)).T,lnIDXUse ,axis=0),shading='gouraud', vmin=minV[device],vmax=maxV[device],cmap=colMap )
                    im4 = ax4.pcolormesh(freqDim,yPlaceHolder,np.repeat(np.reshape(err[4],(len(err[4]),1)).T,lnIDXUse ,axis=0),shading='gouraud', vmin=minV[device],vmax=maxV[device],cmap=colMap )

                    ax0.plot([x[idxFreqAct]]*len(modRange),modRange,'k-',[x[idxFreqActMax]]*len(modRange),modRange,'k-',linewidth=3);
                    ax1.plot([x[idxFreqAct]]*len(modRange),modRange,'k-',[x[idxFreqActMax]]*len(modRange),modRange,'k-',linewidth=3);
                    ax2.plot([x[idxFreqAct]]*len(modRange),modRange,'k-',[x[idxFreqActMax]]*len(modRange),modRange,'k-',linewidth=3);
                    ax3.plot([x[idxFreqAct]]*len(modRange),modRange,'k-',[x[idxFreqActMax]]*len(modRange),modRange,'k-',linewidth=3);
                    ax4.plot([x[idxFreqAct]]*len(modRange),modRange,'k-',[x[idxFreqActMax]]*len(modRange),modRange,'k-',linewidth=3);
##                    plt.yticks(rotation=90)  
                    #plt.tight_layout()
                    plt.show();
#                    
                    ax0.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        labelbottom='off') # labels along the bottom edge are off
                    ax0.yaxis.set(ticks=np.arange(0.25, 1,2), ticklabels=modelLabel)
                    ax0.tick_params(axis='y',rotation=90)
#       
                    ax1.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        labelbottom='off') # labels along the bottom edge are off
                    ax1.yaxis.set(ticks=np.arange(0.25, 1,2), ticklabels=modelLabel[1:])
                    ax1.tick_params(axis='y',rotation=90)
    
                    ax2.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        labelbottom='off') # labels along the bottom edge are off
                    ax2.yaxis.set(ticks=np.arange(0.25, 1,2), ticklabels=modelLabel[2:])
                    ax2.tick_params(axis='y',rotation=90)                   
#                    
                    ax3.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        labelbottom='off') # labels along the bottom edge are off
                    ax3.yaxis.set(ticks=np.arange(0.25, 1,2), ticklabels=modelLabel[3:])
                    ax3.tick_params(axis='y',rotation=90)
                    ax4.yaxis.set(ticks=np.arange(0.25, 1,2), ticklabels=modelLabel[4:])
                    ax4.tick_params(axis='y',rotation=90)

                    plt.show();
                         
                    fig.suptitle('Mean-Squared Error '+'\n Gap:' +str(int(g[device]/1e-6))+ ' $\mu m$, Finger Pairs: '+ str(NFingers[device]*NCombs[device]) +  ' Concentration: '+ str(float(concentration)/1000)+' $mM$ KCl')
                    fig.text(0.01, 0.5, 'Models', va='center', rotation='vertical',fontsize=14)
#                    plt.xlabel('Mean Displacement ($\mu m$)',fontsize=14)
                    plt.xlabel('Frequency (Hz)',fontsize=14)
                    fig.subplots_adjust(left=1.2e-1, bottom=1.5e-1, right=0.85, top=0.9, wspace=None, hspace=None);
                    cax = fig.add_axes([0.88, 0.15,0.05, 0.75])
                    cb = fig.colorbar(im0, cax=cax, ticks=np.linspace(minV[device],maxV[device],11),orientation='vertical');
                    plt.draw();
                    if (concentration == '100'):
                        cb.remove()
                        plt.draw()
#                    plt.tight_layout()
                    #fig.savefig('errorMapPerc_'+device+'_conc'+concentration+'.jpg',format='jpg', dpi=300)
    

def errFullColorMap(b,g,NFingers,NCombs,fitModelString,methodString,deviceString,concentrationString,modelString):
    maxV = {'deviceT3':0.05,'deviceT4':0.05}
    minV = {'deviceT3':-0.05,'deviceT4':-0.05}
    concentrationCol = {'100':'r','1000':'b','10000':'k'};
#    maxV = {'deviceT3':0.2,'deviceT4':0.4}
#    minV = {'deviceT3':-0.2,'deviceT4':-0.4}
    #colMap = cm.cool;#cm.hot;#cm.rainbow; #cm.gray
    colMap = cm.hot;
    freqStart=0;
    freqEnd=24;
    lenFreq = freqEnd-freqStart+1;
    freqArr = np.zeros((25+6));
    freqLow = np.logspace(2,6,25);
    freqHigh = np.logspace(3,7,25);
    freqArr[0:25]=freqLow[0:];
    freqArr[25:]=freqHigh[np.where(freqHigh>1e6)];
    idxFreqActList = [];
    idxFreqMaxList = [];
                        
    for fitModelType in fitModelString:
        for methodUse in methodString:
            thresh = -0.1;
            weights = 1e-10;
            mrkSZ=8;
            
            #Select data to load and initialize relevant parameters
            #device = deviceString[0];
            mnDisplModel=[];
            for device in deviceString:
                runModels = runReducedModels(b[device],g[device],NFingers[device],NCombs[device]);
                modelErr = [];
                mnDisplList=[];
                idxModel=0;
                #Select model from which to obtain error
                idxFreqActList = [];
                idxFreqMaxList = [];
                for modelUse in modelString:  
                    err = [];
                    freqList = [];
                   
                    #freqArr = np.zeros((25+6));
                    freqArrLog = np.zeros((25+6));
                    errArrList = np.zeros((25+6));
                    mnDisplArrTmp = np.zeros((25+6));
                    mnDisplArr = np.zeros((25+6));
                    idxConc=0;
                    #Select concentration of model
                    for concentration in concentrationString:
                        c0NomIn0 = float(concentration)/100; 
                        Vpp = 2.0;  k0 = 1;
                              
                        #Load data
                        data = np.loadtxt('./data/'+device+'/'+concentration+'uMKCl'+'/'+dataFileName); 
                        displData = data[1:,0:]; 
                        mnDispl = np.mean(displData,axis=0)
                        mnDisplList.append(mnDispl);
                        
                        idxUseData = np.where(mnDispl>thresh)[0];
                        mnDispl = mnDispl[idxUseData];
                        displData = data[1:,idxUseData]
                        freqDim = data[0,idxUseData];
                        freqDimLog = np.log10(freqDim)
                        omegaDim = 2*np.pi*freqDim;
                        #x=freqDimLog;
                        x = np.log10(freqArr);
                        #print(x)
                        xMin = 3; xMax = 6.5; #max(x);
                        
                        freqList.append(freqDim);
    #                    xMin = min(mnDispl); xMax = max(mnDispl);
                        #yMin = 0.0; yMax = 1.0;
                        
                        yMin=0; yMax=len(concentrationString)-1;
                        #for modelUse in modelString:
                        modelParams,modelFunc = runModels.reducedModels(modelUse,c0NomIn0,k0,Vpp); 
                        
                        errFunc = modelToDataErr(displData,modelFunc,weights,thresh);
                        
                        #Decide measure of error (L1 or L2 norm)
                        if (fitModelType == 'L2'):
                            errFunc = errFunc.L2NormFreq; 
                        elif (fitModelType == 'L1'):
                            errFunc = errFunc.L1NormFreq; 
                        
                        #Load fit model parameters
                        out = np.load('./data/'+device+'/'+concentration+'uMKCl/'+fitModelType+'_'+methodUse+'_'+modelUse+'.npz');
                        finalParams = out["finalParams"]
                        if modelUse == 'LeakyOxStern':
                            #ROx0,COx0,CStern0,RBulk0,k0
                            ROx = 10**finalParams[0]; 
                            COx = 10**finalParams[1];                           
                            CStern = 10**finalParams[2];
                            RBulk = 10**finalParams[3];
                            modRange = np.array([0,0.5]);
                            freqAct = (1.0/(RBulk*COx))/(2*np.pi);
                            freqActMax = freqAct*10;
                            freqDiff = np.abs(freqArr-freqAct);
                            freqDiffMax = np.abs(freqArr-freqActMax);
                            idxFreqAct = np.where(freqDiff == np.min(freqDiff))[0];
                            #print('IDXFreq',idxFreqAct)
                            idxFreqActMax = np.where(freqDiffMax == np.min(freqDiffMax))[0];
                            print(idxFreqActMax)
                            idxFreqActList.extend(idxFreqAct);
                            idxFreqMaxList.extend(idxFreqActMax);
                            #print(device,concentration,RBulk*COx,(1.0/(RBulk*COx))/(2*np.pi),RBulk/ROx)  
                            
                        #Use loaded parameters to calculate error from data
                        finalParams = list(finalParams); finalParams.insert(0,omegaDim);
                        finalParams = tuple(finalParams);
                        
                        err.append(errFunc(finalParams));
                        #err[idxConc][np.where(np.abs(err[idxConc])>maxV[device])[0]]=maxV[device]*np.sign(err[idxConc][np.where(np.abs(err[idxConc])>maxV[device])[0]]);
                        #print(np.shape(err[idxConc]))
                        if (len(err)==1):
                            #errArr = copy.deepcopy(errArrList);# np.array(err[idxConc]);
                            if (max(freqDim)==1e6):
                                errArrList[0:25]=err[idxConc];
                                errArrList[25:]=err[idxConc][-1];
                                mnDisplArrTmp[0:25]=mnDisplList[idxConc];
                                mnDisplArrTmp[25:]=mnDisplList[idxConc][-1];
                            elif (max(freqDim)==1e7):
                                errArrList[6:]=err[idxConc];
                                errArrList[0:6] = err[idxConc][0];
                                mnDisplArrTmp[6:]=mnDisplList[idxConc];
                                mnDisplArrTmp[0:6] = mnDisplList[idxConc][0];  
                            errArr=copy.deepcopy(errArrList);
                            mnDisplArr=copy.deepcopy(mnDisplArrTmp)
                        else:
                            #errArr = np.vstack([errArr,np.array(err[idxConc])]);
                            
                            if (max(freqDim)==1e6):
                                errArrList[0:25]=err[idxConc];
                                errArrList[25:]=err[idxConc][-1];
                                mnDisplArrTmp[0:25]=mnDisplList[idxConc];
                                mnDisplArrTmp[25:]=mnDisplList[idxConc][-1];
                            elif (max(freqDim)==1e7):
                                errArrList[6:]=err[idxConc];
                                errArrList[0:6] = err[idxConc][0]; 
                                mnDisplArrTmp[6:]=mnDisplList[idxConc];
                                mnDisplArrTmp[0:6] = mnDisplList[idxConc][0]; 
                            
                            errArr = np.vstack([errArr,errArrList]);
                            mnDisplArr = np.vstack([mnDisplArr,mnDisplArrTmp]);
                            
#                    plt.figure();
                       # plt.semilogx(freqDim,err[idxConc])
                        idxConc+=1;
                    modelErr.append(errArr);
                    idxModel+=1;

                if (device=='deviceT3'):
                    plt.figure(1)
                else:
                    plt.figure(2)
#                
#                errArrLeakyOxStern = modelErr[0][0]#/mnDisplArr;                                                         
#                errArrClassic = modelErr[1][0]#/mnDisplArr;                                              
#                errArrLeakyOx  = modelErr[2][0]#/mnDisplArr;               
#                errArrVariableRes  = modelErr[3][0]#/mnDisplArr;               
#                errArrLeakyOxSternVariableRes  = modelErr[4][0]#/mnDisplArr;               
#                
#                modelErrStack = np.vstack(( errArrClassic,errArrVariableRes,errArrLeakyOx,errArrLeakyOxStern,errArrLeakyOxSternVariableRes));
#                print(modelErrStack,np.shape(modelErrStack))
#                fig, axes = plt.subplots(2, 2, sharex='col',sharey='row')
#                axes[0,0].imshow((modelErrStack),aspect='auto',cmap=colMap)

                errArrLeakyOxStern = modelErr[0]                                                       
                errArrClassic = modelErr[1]                                              
                errArrLeakyOx  = modelErr[2]               
                errArrVariableRes  = modelErr[3]               
                errArrLeakyOxSternVariableRes  = modelErr[4]; 
                
                fig, axes = plt.subplots(2,2,figsize=(10,10),sharey=True,sharex=True);#,sharey='row');
                cntSub=0;
                #ax0 = axes[0,0]; ax1 = axes[1,0]; ax2 = axes[2,0]; 
                concUse=0;
                nAcc=4;
                displThrsh=0;
                accBounds = np.array([maxV[device]]*nAcc);
#                if device=='deviceT3':
#                    displThrsh=0;
#                    accBounds = np.array([0.05]*nAcc);
#                else:
#                    displThrsh=0;
#                    accBounds = np.array([0.2]*nAcc);
                #accBounds = np.array([10]*nAcc);
                xPlaceHolder = np.array(np.linspace(0,1,nAcc));
                lines=[];
                #for concUse in range(len(concentrationString)):
                for cntY in range(2):
                    for cntX in range(2): 
                        if (concUse<=3):
                    #for concUse2 in range(len(concentrationString)):
                            mnDisplTmp = mnDisplArr[concUse];
                            errArrClassicTmp=errArrClassic[concUse]; errArrLeakyOxTmp=errArrLeakyOx[concUse];
                            errArrLeakyOxSternTmp=errArrLeakyOxStern[concUse]; errArrVariableResTmp=errArrVariableRes[concUse];
                            errArrLeakyOxSternVariableResTmp=errArrLeakyOxSternVariableRes[concUse];
                            
                            idxFreqActTmp = idxFreqActList[concUse]; idxFreqMaxTmp = idxFreqMaxList[concUse];
                            mnDisplFreqAct = np.array([mnDisplTmp[idxFreqActTmp]]*nAcc)/mnDisplTmp[-1];
                            mnDisplFreqMax = np.array([mnDisplTmp[idxFreqMaxTmp]]*nAcc)/mnDisplTmp[-1];
                            

                            idxMnThrsh = np.where(mnDisplTmp>displThrsh)[0];
                            mnDisplTmp=mnDisplTmp[idxMnThrsh]
                            errArrClassicTmp=errArrClassicTmp[idxMnThrsh]; errArrLeakyOxTmp=errArrLeakyOxTmp[idxMnThrsh];
                            errArrLeakyOxSternTmp=errArrLeakyOxSternTmp[idxMnThrsh]; errArrVariableResTmp=errArrVariableResTmp[idxMnThrsh];
                            errArrLeakyOxSternVariableResTmp=errArrLeakyOxSternVariableResTmp[idxMnThrsh];
                            
                            yPlaceHolder = np.array(np.linspace(-100,100,nAcc));
                            print(mnDisplFreqAct,mnDisplFreqMax,yPlaceHolder,xPlaceHolder)
                            
                            lines = axes[cntX,cntY].plot(mnDisplTmp/mnDisplTmp[-1],errArrLeakyOxSternTmp.T/mnDisplTmp[-1],'-x',mnDisplTmp/mnDisplTmp[-1],errArrClassicTmp.T/mnDisplTmp[-1],'-*',mnDisplTmp/mnDisplTmp[-1],errArrLeakyOxTmp.T/mnDisplTmp[-1],'-s',
                                         mnDisplTmp/mnDisplTmp[-1] ,errArrVariableResTmp.T/mnDisplTmp[-1],'-o',mnDisplTmp/mnDisplTmp[-1] ,errArrLeakyOxSternVariableResTmp.T/mnDisplTmp[-1],'-^',markersize=mrkSZ)
                            axes[cntX,cntY].plot(xPlaceHolder,accBounds,'k-',xPlaceHolder,-accBounds,'k-',
                                mnDisplFreqAct,yPlaceHolder,'m--',mnDisplFreqMax,yPlaceHolder,'m-')
                            #axes[cntX,cntY]
                            if device=='deviceT3':
                                axes[cntX,cntY].set_ylim([-0.15,0.15])
                                axLabel = str(float(concentrationString[concUse])/1000)
                                axes[cntX,cntY].text(0.82,0.1,axLabel + ' mM',fontSize='12')
                            else:
                                axes[cntX,cntY].set_ylim([-0.15,0.15])
                                axLabel = str(float(concentrationString[concUse])/1000)
                                axes[cntX,cntY].text(0.82,0.1,axLabel + ' mM',fontSize='12')
                            #print(concUse)
                        concUse+=1;   
                        #plt.legend(modelString)
                #fig.legend( lines, modelString, 'center', ncol=5)
                fig.legend( lines, modelString, loc=(0.08,0.52), ncol=5)
                #fig.suptitle('Signed Mean-Squared Error of Models'+'\nGap Length: ' +str(int(g[device]/1e-6))+ ' $\mu m$, Finger Pairs: '+ str(NFingers[device]*NCombs[device]),fontsize='20')
                #fig.suptitle('Signed Mean-Squared Error of Models',fontsize='20')
                fig.text(0.5, 0.04, 'Normalized Actuator Displacement', ha='center', fontsize='16')
                fig.text(0.02, 0.5, 'Normalized Displacement Error', va='center', rotation='vertical', fontsize='16')
                #fig.savefig('errorPlots_'+device+'_conc'+concentration+'.jpg',format='jpg', dpi=300,bbox_inches='tight')
                
 

        
def pltModelsAndData(b,g,NFingers,NCombs,fitModelString,methodString,deviceString,concentrationString,modelString):   
    symbString = ['-x','-s','-^','-*','-p'];
    symDict = {'Classic':'-*','LeakyOx':'-s','LeakyOxStern':'-x','VariableRes':'-o','LeakyOxStern+VariableRes':'-^'}
    colorDict={'100':'r','1000':'b','10000':'k'};
    colorDictDev={'deviceT3':'r','deviceT4':'b'};
    freqActDict = {}; freqActOxDict = {}; minTot = 0; maxTot = 0;
    mrkSZ=12;
    lnSz=1
#    #out1D = np.load('1DOptimalResults.npz');
#    out1D = np.load('1000uM_1DOptimalResults.npz');
#    mnDesignClassic1D = out1D['mnDesignClassic1D'];
#    mnDesignLOxStern1D= out1D['mnDesignLOxStern1D'];
#    stdDesignClassic1D=out1D['stdDesignClassic1D'];
#    stdDesignLOxStern1D=out1D['stdDesignLOxStern1D'];
     
    concentrationLegend = [str(float(concentration)/1000)+' $mM$ KCl' for concentration in concentrationString]
    for fitModelType in fitModelString:
        for methodUse in methodString:
            thresh = 0.0;
            weights = 1e-10;
#            fig, axes = plt.subplots(2,1,figsize=(15,15),sharex=True);
            fig, axes = plt.subplots(2,1,figsize=(8,8),sharex=True);  
            cntX=0;
            cntY=0;
            #Select data to load and initialize relevant parameters
            #device = deviceString[0];
           
            for device in deviceString:
                ax=axes[cntX];
                cntX+=1;
                print(b[device])
                runModels = runReducedModels(b[device],g[device],NFingers[device],NCombs[device]); 
                displList = []; modelList = []; #legend=[]
                nFreq = 25; 
                dispMat = np.zeros((len(concentrationString),nFreq));
                freqMat = np.zeros((len(concentrationString),nFreq));
                stdMat = np.zeros((len(concentrationString),nFreq));
                modelDispMat = np.zeros((len(concentrationString),len(modelsToPlot),nFreq));
                cntConc=0;
                
#                fig, ax = plt.subplots()
                lines=[];
                freqActList = [];
                for concentration in concentrationString:
                    c0NomIn0 = float(concentration)/100; 
                    Vpp = 2.0;  k0 = 1;
                    data = np.loadtxt('./data/'+device+'/'+concentration+'uMKCl'+'/'+dataFileName); 
                    displData = data[1:,0:]; 
                    mnDispl = np.mean(displData,axis=0);
                    displData = displData#/mnDispl[-1];
                    mnDispl = mnDispl#/mnDispl[-1];
                    minDispl =  np.min(displData,axis=0);
                    maxDispl =  np.max(displData,axis=0);
                    
                    if (np.min(minDispl)<=minTot):
                        minTot = np.min(minDispl);
                    elif (np.max(maxDispl)>=maxTot):
                        maxTot = np.max(maxDispl);  
                    displRange = np.linspace(0,1,nFreq);#np.linspace(minTot,maxTot,nFreq);
                    stdDispl = np.std(displData,axis=0);
                    #print(stdDispl)
                    #print([np.abs(maxDispl-mnDispl),np.abs(minDispl-mnDispl)])
                    freqDim = data[0,0:];
                    omegaDim = 2*np.pi*freqDim;
                    cntModel=0;
                    dispMat[cntConc] = mnDispl;#np.mean(displData,axis=0); 
                    freqMat[cntConc] = freqDim;
                    stdMat[cntConc] = stdDispl;#np.std(displData,axis=0);
                    cntModel=0;
#                    ax.semilogx(freqDim,displData.T,'o')
                    #ax.errorbar(freqDim,mnDispl/mnDispl[-1],yerr=2*stdDispl,fmt=colorDictDev[device]+'o',markersize=mrkSZ);
                   # ax.errorbar(freqDim,mnDispl,yerr=[np.abs(maxDispl-mnDispl),np.abs(minDispl-mnDispl)],fmt=colorDict[concentration]+'o',markersize=3);
                    ax.set_xscale('log')
                    lines.extend(ax.semilogx(freqDim,mnDispl/mnDispl[-1],colorDictDev[device]+'o',markersize=mrkSZ,markerfacecolor='None'))
                    legend=[];
                    for modelUse in modelsToPlot:
                        #print("Vpp",Vpp)
                        modelParams,modelFunc = runModels.reducedModels(modelUse,c0NomIn0,k0,Vpp); 
                        out = np.load('./data/'+device+'/'+concentration+'uMKCl/'+fitModelType+'_'+methodUse+'_'+modelUse+'.npz');
                        finalParams = out["finalParams"]
                        #print(modelUse,10**finalParams)
                        if modelUse == 'LeakyOxStern':
                            #ROx0,COx0,CStern0,RBulk0,k0
                            ROx = 10**finalParams[0]; 
                            COx = 10**finalParams[1];                           
                            CStern = 10**finalParams[2];
                            RBulk = 10**finalParams[3];
                            print(modelUse,device,concentration,'BulkOx',RBulk*COx,'Ox Ox',ROx*COx,'BulkStern',RBulk*CStern,'OxStern',ROx*CStern)
                            freqActDict[concentration] = np.array([(1.0/(RBulk*COx))/(2*np.pi)]*nFreq);
                            freqActList.append(freqActDict[concentration])
                            #print(freqActDict)
                        elif (modelUse== 'Classic'):
                            pass
                            COx = 10**finalParams[0];
                            RBulk = 10**finalParams[1];
                            #freqActDict[concentration] = np.array([(1.0/(RBulk*COx))/(2*np.pi)]*nFreq);
                            #freqActList.append(freqActDict[concentration])
                            #print(modelUse,device,concentration,RBulk*COx,'RC Freq',1.0/(RBulk*COx))
                        if (modelUse == 'Classic' or modelUse == 'LeakyOx' or modelUse == 'LeakyOxStern' or modelUse == 'LeakyOxStern+VariableRes'):
                            finalParams = list(finalParams); finalParams.insert(0,omegaDim);
                            finalParams = tuple(finalParams)
                            modelDispMat[cntConc,cntModel,0:] = modelFunc(*finalParams); 
                           # freqActDict[concentration] = np.array([(1.0/(RBulk*COx))/(2*np.pi)]*nFreq);
                           # freqActList.append(freqActDict[concentration])
                            cntModel = cntModel+1;
                    cntConc=cntConc+1
                #legend = np.append(legend,concentrationLegend)
                #legend = np.append(legend,['gap: ' + str(g[device]/1e-6)+' $\mu m$ Fingers: '+ str(NFingers[device]*NCombs[device]) + ', ' + conc for conc in concentrationLegend]);
                legend = np.append(legend,str(concentrationLegend[0] + ' Data'));
                axLabel = 'gap: ' + str(g[device]/1e-6)+' $\mu m$\nFingers: '+ str(NFingers[device]*NCombs[device]);
                #ax.text(1e2,0.8,axLabel,fontSize='18')
                cntConc = 0;
                
                for concentration in concentrationString:
                    cntModel=0;
                    for modelUse in modelsToPlot:
##                        pass
##                        pass
##                        if modelUse == 'Classic':
##                            ax.semilogx(freqMat[cntConc],modelDispMat[cntConc,cntModel,0:]/modelDispMat[cntConc,cntModel,-1],colorDictDev[device]+symDict[modelUse])
##                            ax.semilogx(freqMat[cntConc],modelDispMat[cntConc,cntModel,0:]/modelDispMat[cntConc,cntModel,-1],colorDictDev[device]+symDict[modelUse])
##                            lines = ax.semilogx(freqMat[cntConc],modelDispMat[cntConc,cntModel,0:]/modelDispMat[cntConc,cntModel,-1],colorDictDev[device]+symDict[modelUse],markersize=mrkSZ)
                            lines.extend(ax.semilogx(freqMat[cntConc],modelDispMat[cntConc,cntModel,0:]/modelDispMat[cntConc,cntModel,-1],colorDictDev[device]+symDict[modelUse],markersize=mrkSZ,linewidth=lnSz))
                            legend = np.append(legend,modelsToPlot[cntModel]+' '+concentrationLegend[cntConc])
                            cntModel=cntModel+1;
                    cntConc=cntConc+1;
                matplotlib.rc('xtick', labelsize=18) 
                matplotlib.rc('ytick', labelsize=18) 
                #legend = np.append(legend,modelsToPlot)  
                #plt.legend(legend,loc=2)
                #fig.legend(legend,loc=2)
                #plt.title(device)
                #plt.xlabel('Frequency (Hz)',fontsize='14')
                #plt.ylabel('Displacement ($\mu m$)')
                #plt.ylabel('Normalized Displacement',fontsize='14')
                for concentration in concentrationString:
                    if (concentration=='1000'):                       
                        #pass
                        #print(yPlace,mnDesignLOxStern1D[0:,0],np.shape(yPlace),np.shape(mnDesignLOxStern1D[0:,0]))
                        ax.plot(freqActDict[concentration],displRange,'m--',freqActDict[concentration]*10,displRange,'m-')
                        #yPlace = np.array([1]*len(mnDesignLOxStern1D[0:,0]));
                        #ax.plot(10**mnDesignClassic1D[0:,0],yPlace-0.2,'r*',markersize=mrkSZ)
                        #ax.plot(10**mnDesignLOxStern1D[0:,0],yPlace,'b*',markersize=mrkSZ)
        print("legend",legend,"lines",lines)       
        fig.text(0.5, 0.01, 'Frequency (Hz)', ha='center', fontsize='16')
        fig.text(0.01, 0.5, 'Normalized Actuator Displacement', va='center', rotation='vertical', fontsize='16')
        #fig.legend( lines, legend, 'lower right', ncol=1)
        #plt.legend(legend,fontsize='15',loc='lower right')
        fig.legend( lines, legend, loc=(0.15,0.53), ncol=3)
        fig.savefig('dataSpeed_'+device+'_conc'+'.jpg',format='jpg', dpi=300,bbox_inches='tight')
        return freqActList
                        


methodString = ['GCPSO'];
fitModelString = ['L2']; 
dataFileName = '4Vpp_Device12.dat';               
concentrationString = ['100'];#,'500','1000','5000','10000'];
#concentrationString = ['500'];
modelString = ['LeakyOxStern','Classic'];#,'LeakyOx','VariableRes','LeakyOxStern+VariableRes'];
label = range(len(modelString));
modelLabel = ['Classic','LeakyOx\nStern'];#,'Variable\nRes','LeakyOxStern\n+VarRes'];
modelsToPlot =['Classic','LeakyOxStern'];#,'LeakyOxStern+VariableRes']

gArr = np.array([5.0e-6,2e-6]); 
bArr = np.array([15e-6,15e-6]); 
NFingersArr = np.array([25,25]); 
NCombsArr = np.array([4,8]);
#deviceString = ['deviceT3','deviceT4']; 
deviceString = ['deviceT3','deviceT4'];
  
b={}; g={}; NFingers={}; NCombs={};
for l in range(len(deviceString)):
    b[deviceString[l]] = bArr[l];
    g[deviceString[l]] = gArr[l]; 
    NFingers[deviceString[l]] = NFingersArr[l];
    NCombs[deviceString[l]] = NCombsArr[l];

noPlotModel = 0;
plotModel = 1;
plotModelIndc = plotModel;  


#errColorMap(b,g,NFingers,NCombs,fitModelString,methodString,deviceString,concentrationString,modelString);     
#freqActConc=pltModelsAndData(b,g,NFingers,NCombs,fitModelString,methodString,deviceString,concentrationString,modelString)         
errFullColorMap(b,g,NFingers,NCombs,fitModelString,methodString,deviceString,concentrationString,modelString);              
                    