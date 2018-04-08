import numpy as np
from scipy.integrate import odeint
import matplotlib as mplib
import matplotlib.pyplot as plt
from reducedCircuitModels import circuitModel_LinCap
from reducedCircuitModels import circuitModel_NonLinCap
from reducedCircuitModels import circuitModel_NonLinCap_VarRes
from reducedCircuitModels import circuitModelNonLinCapROxVarRes
from reducedCircuitModels import  circuitModel_NonLinCap_InhomResPython
#from reducedCircuitModels import circuitModel_NonLinCap_InhomResPython

import time
from scipy.interpolate import interp1d
class implementationClass():
    def implementCircuitModel(self,initConditions,circuitParams,model,time,dx,eps,frequency,delta_phi0):
        epsilonR=circuitParams.epsilonR
        epsilonOx=circuitParams.epsilonOx
        epsilonStern=circuitParams.epsilonStern
        C0=circuitParams.C0
        Cox=circuitParams.Cox
        Cstern=circuitParams.Cstern
        #CB=circuitParams.CB
        R=circuitParams.R
        
        cInit=initConditions.cInit
        c0=cInit[0]/2
        q0Init=initConditions.q0Init
        v0Init=initConditions.v0Init
        vOxInit=initConditions.vOxInit
        vSternInit=initConditions.vSternInit
        vBInit=initConditions.vBInit   
        c0Init=initConditions.c0Init   
        R0Init=initConditions.R0Init  
        
        simulationType = model.simulationType
        Normal = model.NormalCircuit
        Reduced = model.ReducedCircuit
        circuitModel = model.circuitModel
        LinCapModel = model.LinCapModel
        NonLinCapModel = model.NonLinCapModel
        NonLinCapVarResModel = model.NonLinCapVarResModel
        NonLinCapInhomResModel = model.NonLinCapInhomResModel
        NonLinCapVarResROxModel = model.NonLinCapVarResROxModel

        if circuitModel==LinCapModel:
            iInit=([0,(delta_phi0-(v0Init+vOxInit+vSternInit))/R,v0Init,vOxInit,vSternInit,delta_phi0-(v0Init+vOxInit+vSternInit)])
            i=odeint(circuitModel_LinCap,iInit,time,args=(R,C0,Cox,Cstern,frequency,delta_phi0))
            charge=i[:,0]
            current=i[:,1]
            vEdl=i[:,2]
            vOx=i[:,3]
            vStern=i[:,4]
            vBulk=i[:,-1]
            m=1./dx  
            c=np.ones((len(vBulk),m))
            c=2*c
            cTot=(0.5*(c[:,0]+c[:,-1])+np.sum(c[:,1:-1],axis=1))*dx  

            R=R*np.ones(np.shape(vBulk))
            timeBulk=time
        elif circuitModel==NonLinCapModel:
            iInit=([vOxInit*Cox,(delta_phi0-2*(v0Init+vOxInit+vSternInit))/R,v0Init,vOxInit,vSternInit,delta_phi0-2*(v0Init+vOxInit+vSternInit)])
            i=odeint(circuitModel_NonLinCap,iInit,time,args=(R,C0,Cox,Cstern,frequency,delta_phi0))
            charge=i[:,0]
            current=i[:,1]
            vEdl=i[:,2]
            vOx=i[:,3]
            vStern=i[:,4]
            vBulk=i[:,-1]
            m=1./dx  
            c=cInit[0]*np.ones((len(vBulk),m))
            c=2*c
            cTot=(0.5*(c[:,0]+c[:,-1])+np.sum(c[:,1:-1],axis=1))*dx  
            R=R*np.ones(np.shape(vBulk))       
            timeBulk=time   
        elif circuitModel==NonLinCapVarResModel:
#            #nFreq=len(frequency)
#            iInit=([q0Init,(vBInit-(v0Init+vOxInit+vSternInit))/R,v0Init,vOxInit,vSternInit,c0,R,vBInit-(v0Init+vOxInit+vSternInit)]) 
#            iInit=np.append(q0Init,(vBInit-(v0Init+vOxInit+vSternInit))/R)#([0,delta_phi0/R,0,R,cInit,eps,delta_phi0])
#            iInit=np.append(iInit,v0Init)
#            iInit=np.append(iInit,vOxInit)
#            iInit=np.append(iInit,vSternInit)
#            iInit=np.append(iInit,c0Init)
#            iInit=np.append(iInit,R0Init)
#            iInit=np.append(iInit,vBInit-(v0Init+vOxInit+vSternInit))
            iInit=([q0Init,(vBInit-(v0Init+vOxInit+vSternInit))/R,v0Init,vOxInit,vSternInit,c0,R,vBInit-(v0Init+vOxInit+vSternInit)])
#            charge,current,vEdl,vOx,vStern,R,c,vBulk=odeint(circuitModel_NonLinCap_VarRes,iInit,time,args=(C0,Cox,Cstern,frequency,delta_phi0));
            i=odeint(circuitModel_NonLinCap_VarRes,iInit,time,args=(C0,Cox,Cstern,frequency,delta_phi0))
            #print("i",i,np.shape(i))
            [charge,current,vEdl,vOx,vStern,c,R,vBulk] = [i[:,m] for m in range(len(iInit))] 
            
#            charge=i[:,0]
#            current=i[:,1]
#            vEdl=i[:,2]
#            vOx=i[:,3]
#            vStern=i[:,4]
#            R=i[:,5]
#            c=i[:,6]
#            vBulk=i[:,-1]  
#            m=int(1./dx);
#            c=np.multiply(np.reshape(c,(len(vBulk),1)),np.matrix(np.ones((len(vBulk),m))))
#            c=2*c
#            cTot=(0.5*(c[:,0]+c[:,-1])+np.sum(c[:,1:-1],axis=1))*dx  
            c=2*c;
            cTot=np.array([2.0]);
            #print("time",time,len(time))
            timeBulk=time
            #plt.plot(time,R)
            # plt.plot(time,vBulk)
        elif circuitModel==NonLinCapVarResROxModel:
            ROx = circuitParams.ROx;
            iInit=([q0Init,(vBInit-(v0Init+vOxInit+vSternInit))/R,v0Init,vOxInit,vSternInit,c0,R,vBInit-(v0Init+vOxInit+vSternInit)])
            #iInit=([0,(vBInit-(v0Init+vOxInit+vSternInit))/R,v0Init,vOxInit,vSternInit,c0,R,vBInit-(v0Init+vOxInit+vSternInit)])
            i=odeint(circuitModelNonLinCapROxVarRes,iInit,time,args=(C0,Cox,Cstern,ROx,frequency,delta_phi0));
            [charge,current,vEdl,vOx,vStern,c,R,vBulk] = [i[:,m] for m in range(len(iInit))] 
#            charge=i[:,0]
#            current=i[:,1]
#            vEdl=i[:,2]
#            vOx=i[:,3]
#            vStern=i[:,4]
#            R=i[:,5]
#            c=i[:,6]
#            vBulk=i[:,-1] 
#            m=int(1./dx);
#            c=np.multiply(np.reshape(c,(len(vBulk),1)),np.matrix(np.ones((len(vBulk),m))))
            c=2*c
            cTot=np.array([2.0]);#(0.5*(c[:,0]+c[:,-1])+np.sum(c[:,1:-1],axis=1))*dx  
            timeBulk=time
        elif circuitModel==NonLinCapInhomResModel:
            m=int(1.0/dx)
            n=int(1.0/dx)
            cInit=(2*c0*np.ones((m,1)))
            x=np.linspace(0,1,m)
            diags=np.asarray([1.0,2.0,1.0])
            A=np.zeros((n,m),diags.dtype)
            b=np.zeros((n,1),diags.dtype)   
            for z, v in enumerate((1.0,-2.0,1.0)):
                np.fill_diagonal(A[1:-1,z:],v) 
            A[0,0]=-2.0
            A[0,1]=2.0
            A[-1,-1]=-2.0
            A[-1,-2]=2.0
            A=A/(dx**2)
            A=np.matrix(A)
            b[0]=-2
            b[-1]=-2
            iInit=np.append(0,(vBInit-(v0Init+vOxInit+vSternInit))/R)#([0,delta_phi0/R,0,R,cInit,eps,delta_phi0])
            iInit=np.append(iInit,v0Init)
            iInit=np.append(iInit,vOxInit)
            iInit=np.append(iInit,vSternInit)
            iInit=np.append(iInit,R)
            iInit=np.append(iInit,cInit)
            iInit=np.append(iInit,vBInit-(v0Init+vOxInit+vSternInit))
            #print iInit
            i=odeint(circuitModel_NonLinCap_InhomResPython,iInit,time,args=(dx,eps,C0,Cox,Cstern,frequency,delta_phi0,A,b))
            charge=i[:,0]
            current=i[:,1]
            vEdl=i[:,2]
            vOx=i[:,3]
            vStern=i[:,4]
            R=i[:,5]
            vBulk=i[:,-1]               
            c=i[:,6:-1]
            cTot=(0.5*(c[:,0]+c[:,-1])+np.sum(c[:,1:-1],axis=1))*dx  
            timeBulk=time
        return [charge, current, vEdl, vOx, vStern, c, cTot, R, vBulk,timeBulk]

    def implementSimulation(self,simulationParams,delta_phi0,frequency,dt,dx,t0,tEnd):
        
         ### Simulation Parameters ###
        #Universal Constants
        kB=simulationParams.kB #Boltzmann Constant
        epsilon0=simulationParams.epsilon0 #Permitivitty of Free Space
        eCharge=simulationParams.eCharge #(C)
        NA = simulationParams.NA #Avogadro's Number(1/mol)
        F = eCharge*NA #Faraday's Constant (C/mol)
        R=simulationParams.R #Gas Constant(J mol/K)
        Di = simulationParams.Di #Diffusivity (m2/s)
        
        #System Constants
        T=simulationParams.T #Temperature(K)
        epsilonR=simulationParams.epsilonR #Relative Permitivitty of Water
        epsilonOx=simulationParams.epsilonOx #Relative Permitivitty of Oxide
        epsilonStern=simulationParams.epsilonStern #Relatie Permitivitty of Stern Layer
        V_T=simulationParams.V_T #Thermal Voltage (V)
        L=simulationParams.L #Domain Separation Length
        eps =simulationParams.eps #Dimensionless Debye Length
        lambda_d = L*eps #Debye Length
        lambda_Stern=simulationParams.lambda_Stern
        lambda_Ox=simulationParams.lambda_Ox
        c0 = (epsilonR*epsilon0*kB*T)/(2*(eCharge**2)*(lambda_d**2)) #Nominal Bulk Concentration (#/m3)
        
    
        ### species ###
        # number of species
        num_species = simulationParams.num_species
        # valences
        z_i = simulationParams.z_i
        # diffusivities
        D_i = simulationParams.D_i
    
        
        ########Boundary Condition Parameters########
        oxideLayer=simulationParams.oxideLayer #Oxide Layer Boolean
        sternLayer=simulationParams.sternLayer #Stern Layer Boolean
        Cox=simulationParams.Cox #Dimensionless capacitance of oxide
        Cstern=simulationParams.Cstern #Dimensionless capacitance of Stern Layer
        species_bc_types = [["no flux", "no flux"], ["no flux", "no flux"]]
        species_bc_vals = [(None, None), (None, None)]
    
        #####Spatial and Temporal Discretizatons#######
        time_space=np.linspace(t0,tEnd-dt,(tEnd-t0)/dt) #Time Discretization
        N = 1./dx # total number of mesh points

        ############ Initialize Mesh and System States ###################################
        ### Form Mesh ###
        minMesh=simulationParams.minMesh
        mesh = Mesh(N, "tanh", minMesh)
        
        x_uniform_cent=mesh.x_uniform_cent
        x_cent = mesh.x_cent
        x_stag = mesh.x_stag
        dzdx_cent=mesh.dzdx_cent
        dzdx_stag=mesh.dzdx_stag
        
        ### Initialize Fields ###
        # divide by |z_i| in the initial concentration to ensure net electroneutrality
        # as long as the number of anionic and cationic species is equal
        c_star_i = [np.ones(N)/np.abs(z_i[i]) for i in range(num_species)]
        c_n_i = [np.copy(c_star) for c_star in c_star_i]
        c_nm1_i = [np.copy(c_star) for c_star in c_star_i]
    
    
        ### Initialize Solvers ###
        # Poisson Solver
        phi_solver = poisson_solver.PoissonSolver(mesh, eps)
        phi_star = phi_solver.solve(c_star_i, z_i, delta_phi0)
    
        # Species Solver
        c_solver = implicit_species_solver.ImplicitSpeciesSolver(mesh, num_species,
                        species_bc_types, species_bc_vals, ["dirichlet", "dirichlet"],
                        eps, z_i, D_i)
    
    
        q_I = np.zeros((tEnd/dt,1))
        q_J_i = np.zeros((tEnd/dt,1))
        cPlus = np.zeros((tEnd/dt,N))
        cMinus = np.zeros((tEnd/dt,N))
        c_Tot = np.zeros((tEnd/dt,1))
        phi = np.zeros((tEnd/dt,N))
        dVdx_i = np.zeros((tEnd/dt,1))
        delta_phi_i = np.zeros((tEnd/dt,1))
        J_i = np.zeros((tEnd/dt,1))
            
            
    
        iter_count_l = []
        surfCharge=0.0 #Initialize Surface Charge Density
        #Simulation loop
        print "Full Simulation Model: "
        for time_i in xrange(int(tEnd/dt)):
            delta_phi = delta_phi0*np.cos(frequency*time_i*dt) #Applied potential
            delta_phi_i[time_i] = delta_phi #Store applied potential
            delta_phiRight=delta_phi #Initialize Right Bounday Potential
            delta_phiLeft=0 #Initialize Left Boundary Potential
            
            #Modify boundary potentials based on presence of oxide and stern layers
            if oxideLayer==1:
                dphidxBnd=(surfCharge)/(Cox*(lambda_Ox/L))    
                delta_phiRight=delta_phiRight-dphidxBnd*(lambda_Ox/L)
                delta_phiLeft=delta_phiLeft+dphidxBnd*(lambda_Ox/L)
            if sternLayer==1:
                dphidxBnd=(surfCharge)/(Cstern*(lambda_Stern/L))  
                delta_phiRight=delta_phiRight-dphidxBnd*(lambda_Stern/L)
                delta_phiLeft=delta_phiLeft+dphidxBnd*(lambda_Stern/L)  
            
            delta_norm = 1. #Initialize norm
            iter_count = 0 #Initialize iteration
            # perform Newton iteration
            while delta_norm > 1e-9:
                iter_count = iter_count + 1
                phi_star = phi_solver.solve(c_star_i, z_i, delta_phiRight-delta_phiLeft)
                delta = c_solver.newton_iterate(dt, c_star_i, c_n_i, c_nm1_i, phi_star, phi_bc_vals=(delta_phiLeft, delta_phiRight))
                delta_norm = np.sqrt(np.mean(np.power(delta, 2.)))
                
            tempQ=np.zeros(np.shape(phi_star)) #Initialize array for storing charge distribution before calculatnig total charge
            tempC=np.zeros(np.shape(phi_star)) #Initialize array for storing concentration distribution before calculatnig total charge
            for i in range(num_species):
                c_nm1_i[i] = np.copy(c_n_i[i])
                c_n_i[i] = np.copy(c_star_i[i])
                tempQ = tempQ + c_n_i[i]*z_i[i]*(1./(mesh.dzdx_cent))
                tempC = tempC + c_n_i[i]*(1./(mesh.dzdx_cent))
            iter_count_l.append(iter_count)
            
            #Store potential profiles
            phi[time_i][0:]=phi_star
            
            #Store concentrations and surface charge densities
            q_I[time_i]=(0.5*(tempQ[0]+tempQ[N/2-1])+np.sum(tempQ[1:N/2-2])) #Calculate total surface charge density
            surfCharge=q_I[time_i] #Store surface charge density for use in next time step to modify boundary conditions
            
            c_Tot[time_i]=(0.5*(tempC[0]+tempC[N/2-1])+np.sum(tempC[1:N/2-2])) #Calculate total concentration 
            cPlus[time_i,0:]=c_n_i[0] #Positive concentration profile
            cMinus[time_i,0:]=c_n_i[1] #Negative concentration profile

            #Calculate current flux using information at center of the domain
            dphi_dx = (phi_star[N/2]*V_T-phi_star[N/2-1]*V_T)*(mesh.dzdx_stag[N/2])/L #Potential gradient at the center of the domain
            dcplus_dx= c0*(c_n_i[0][N/2]-c_n_i[0][N/2-1])*mesh.dzdx_stag[N/2-1]/L #Positive Concentration gradient at the center of the domain
            dcminus_dx= c0*(c_n_i[1][N/2]-c_n_i[1][N/2-1])*mesh.dzdx_stag[N/2-1]/L #Negative Concentration gradient at the center of the domain
            xCent=[L*mesh.x_cent[N/2-1], L*mesh.x_cent[N/2]]; #Store two points on either side of the domain center
            cCent=np.array([[c_n_i[0][N/2-1], c_n_i[0][N/2]], [c_n_i[1][N/2-1], c_n_i[1][N/2]]]); #Store concentrations on either side of the domain center
            cInterp=interp1d(xCent,cCent) #Interpolation of concentration profile at the center
            cCent=cInterp(L*mesh.x_stag[N/2]) #Obtain concentration at domain center           
            #Calculate dimensional current flux
            J_i[time_i] = Di*eCharge*z_i[0]*(dcplus_dx + z_i[0]*eCharge*c0*cCent[0]*dphi_dx/(kB*T)) + Di*eCharge*z_i[1]*(dcminus_dx + z_i[1]*eCharge*c0*cCent[1]*dphi_dx/(kB*T))
            J_i[time_i] = L*J_i[time_i]/(Di*eCharge*c0) #Non-dimensionalize the current flux
            if (time_i>0):
                    q_J_i[time_i-1][0] = dt*(0.5*(J_i[0]+J_i[time_i])+np.sum(J_i[1:time_i-1])) #Calculate charge density by integrating current flux
        
        
        I = (3*q_I[0][2:]-4*q_I[0][1:-1]+q_I[0][0:-2])/(2*dt) #Calculate current flux by differentiating charge density
        
        return [q_I, q_J_i, I, J_i, cPlus, cMinus, c_Tot, phi, delta_phi_i, x_uniform_cent, x_cent, x_stag, dzdx_cent, dzdx_stag, time_space]

        
            
