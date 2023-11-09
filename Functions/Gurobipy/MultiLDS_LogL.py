class LogL_Gurobi1(object):
    """Clustering Estimator based on Gurobi
    """
    
    def __init__(self, **kwargs):
        super(LogL_Gurobi1, self).__init__()

    def estimate(self, data, K):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        data: array
            Variable seen as input
        K: int
            Variable seen as number of clusters

        Returns
        -------
        X_predict: array
            regression predict values of X or residuals
        obj: num
            Objective value in optima
        """
        
        S = len(data)
        
        # Create a new model
        e = gp.Env()
        e.setParam('TimeLimit', 3*60)
        m = gp.Model(env=e)
        

        # Obs dim
        L = len(np.transpose(data))
        data =[((l,s),np.transpose(data).iloc[l,s]) for l in range(L) for s in range(S)]
        Y = tupledict(data)

        #L = tupledict({(0,0):1, (0,1): 1, (0,2): 1, (0,3): 0, (0,4): 0, (0,5): 0})
        #N=T*nL
        #print('T is ' + str(T)+', Groups number is ' + str(nL))

        # Set hidden_state_dim(N0),
        N0 = 2        
        T = 4

        # Create variables
        # L = m.addVars(1, nL, name="L", vtype='B')
        F = m.addVars(N0,N0, name="F",vtype='C')
        X = m.addVars(N0,S,(T+1), name="X", vtype='C')
        Sigma = m.addVars(N0,N0, name="Sigma", vtype='C')
        G = m.addVars(L,N0,(T+1), name="G",vtype='C')      
        Y = m.addVars(L,S,T, name="Y", vtype='C')
        V = m.addVars(L,S,T, name="V", vtype='C') 
        K = m.addVars(N0,L,(T+1), name="K",vtype='C')
        P = m.addVars(N0,N0,(T+1), name="P",vtype='C')
        
        # Dummy variables
        li = m.addVars(L,S,T, name="li", vtype='C')
        Res = m.addVars(L,S,T, name="Res", vtype='C')
        ResS = m.addVars(L,L,T, name="ResS", vtype='C')
        w = m.addVars(N0,N0,(T+1), name="w", vtype='C')
        ww = m.addVars(N0,N0,(T+1), name="ww", vtype='C')
        GP = m.addVars(L,N0,T, name="GP", vtype='C')
        GPG = m.addVars(L,L,T, name="GPG", vtype='C')
        GPG_I = m.addVars(L,L,T, name="GPG_I", vtype='C')
        FPG = m.addVars(N0,L,T, name="FPG", vtype='C')
        
        ############# Def I
        # I[l,l,t]
        ############# Initiations
        
        
        #model.addVars(2, 3)
        #model.addVars([0, 1, 2], ['m0', 'm1', 'm2'])
        #print("This model has",n0*n0* 2+n0*nL +n0*(T+1)* 2+n0*T* 2+T*nL* 3+n1*n1* 2+n1*nL +n1*(T+1)* 2 +n1*T* 2,"decision variables.")

        obj = gp.quicksum(ResS[l,l,t]*GPG_I[l,l,t] for l in range(L) for t in range(T))
        # obj += gp.quicksum(log(GPG[l,l,t]+V[l,l,t]) for l in range(L) for t in range(T)) 
    

        m.setObjective(obj, GRB.MINIMIZE)


        # AddConstrs
        ## Res_Square
        m.addConstrs((li[l,s,t] == G[l,n_,t]*X[n_,s,t]) for l in range(L) for n_ in range(N0) for t in range(T) for s in range(S))   
        m.addConstrs((Res[l,s,t] == Y[l,s,t]- li[l,s,t]) for l in range(L) for t in range(T) for s in range(S))
        m.addConstrs((ResS[l,l,t] == Res[l,s,t]*Res[l,s,t]) for l in range(L) for t in range(T) for s in range(S))
        
        ## P
        m.addConstrs((w[n_,n_,t+1] == F[n_,n_]- K[n_,l,t+1]*G[l,n_,t+1]) for l in range(L) for n_ in range(N0) for t in range(T))   
        m.addConstrs((ww[n_,n_,t+1] == w[n_,n_,t+1]*P[n_,n_,t]) for l in range(L) for n_ in range(N0) for t in range(T))   
        m.addConstrs((P[n_,n_,t+1] == ww[n_,n_,t+1]*w[n_,n_,t+1]) for l in range(L) for n_ in range(N0) for t in range(T))   
        
        ## K
        m.addConstrs((GP[l,n_,t] == G[l,n_,t]*P[n_,n_,t]) for l in range(L) for n_ in range(N0) for t in range(T))   
        # G[l,b,t] = GT[n_,l,t]
        m.addConstrs((GPG[l,l,t] == GP[l,n_,t]*G[l,n_,t]) for l in range(L) for n_ in range(N0) for t in range(T))   
        ## (GPG+V)^(-1)
        m.addConstrs((I[l,l,t] == GPG_I[l,l,t]*(GPG[l,l,t]+V[l,l,t])) for l in range(L) for n_ in range(N0) for t in range(T))   
        m.addConstrs((FPG[n_,l,t] == F[n_,n_]*P[n_,n_,t]*GT[n_,l,t]) for l in range(L) for n_ in range(N0) for t in range(T))   
        m.addConstrs((K[n_,l,t] == FPG[n_,l,t]*GPG_I[l,l,t]) for l in range(L) for n_ in range(N0) for t in range(T))   
        
        ## X
        m.addConstrs((X[n_,s,(t+1)] == F[n_,n_]*X[n_,s,t]+ K[n_,l,t]*Res[l,s,t]) for l in range(L) for n_ in range(N0) for t in range(T) for s in range(S))   
        
        # m.addConstrs((phi0[n_,(t+1)] == G0[n_,n_]*phi0[n_,t] + p0[n_,t]) for n_ in range(n0) for t in range(T))               
        # m.addConstrs((X0[l,t] == L[0,l] * X[l,t]) for l in range(nL) for t in range(T))
        # m.addConstrs((v0[l,t] == F0[l,n_]*phi0[n_,(t+1)]) for l in range(nL) for n_ in range(n0) for t in range(T))   
        # m.addConstrs((f0[l,t] == L[0,l] * v0[l,t] + L[0,l] * q[l,t]) for l in range(nL) for t in range(T))  
        m.update()

        # Solve it!
        m.Params.NonConvex = 2
        #m.setParam('OutputFlag', 0)

        
        m.optimize()
        # if m.objVal <= 1:
        print(f"Optimal objective value: {m.objVal}")

        print(f"m.status is " + str(m.status))
        print(f"GRB.OPTIMAL is "+ str(GRB.OPTIMAL))

        if m.status == GRB.Status.OPTIMAL:
            print(f"THIS IS OPTIMAL SOLUTION")
        else:
            print(f"THIS IS NOT OPTIMAL SOLUTION")

        
        #if m.status == GRB.Status.OPTIMAL:
        #    print(f"THIS IS OPTIMAL SOLUTION")
        #    print(f"Optimal objective value: {m.objVal}")
            
        # Print solution
        print(f"Optimal L value:")
        print(np.array([m.getAttr("x",L)[h] for h in m.getAttr("x",L)]))  
        '''
        print(f"Optimal G0 value:")
        print(np.array([m.getAttr("x",G0)[h] for h in m.getAttr("x",G0)]).reshape(n, n))  
        print(f"Optimal G1 value:")
        print(np.array([m.getAttr("x",G1)[h] for h in m.getAttr("x",G1)]).reshape(n, n)) 
        print(f"Optimal F0 value:")
        print(np.array([m.getAttr("x",F0)[h] for h in m.getAttr("x",F0)]).reshape(N0, n))         
        print(f"Optimal F1 value:")
        print(np.array([m.getAttr("x",F1)[h] for h in m.getAttr("x",F1)]).reshape(N1, n))         
        print(f"Optimal phi0 value:")
        print(np.array([m.getAttr("x",phi0)[h] for h in m.getAttr("x",phi0)]).reshape(n, (T+1)))
        print(f"Optimal phi1 value:")
        print(np.array([m.getAttr("x",phi1)[h] for h in m.getAttr("x",phi1)]).reshape(n, (T+1)))
        print(f"Optimal f0 value:")
        print(np.array([m.getAttr("x",f0)[h] for h in m.getAttr("x",f0)]).reshape(N0, T)) 
        print(f"Optimal f1 value:")
        print(np.array([m.getAttr("x",f1)[h] for h in m.getAttr("x",f1)]).reshape(N1, T))         
    '''
        #return

class MultiLDS_Gurobi2(object):
    """Clustering Estimator based on Gurobi
    """
    
    def __init__(self, **kwargs):
        super(MultiLDS_Gurobi2, self).__init__()

    def estimate(self, data, K):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        data: array
            Variable seen as input
        K: int
            Variable seen as number of clusters

        Returns
        -------
        X_predict: array
            regression predict values of X or residuals
        obj: num
            Objective value in optima
        """
        
        T = len(data)
        
        # Create a new model
        # e = gp.Env()
        # e.setParam('TimeLimit', 3*60)
        # m = gp.Model(env=e)
        m = gp.Model()
        

        # Obs dim L,S,T
        L = len(np.transpose(data))
        #data =[((l,0,T),np.transpose(data).iloc[l,t]) for l in range(L) for t in range(T)]
        da = []
        for t in range(T):
            for l in range(L):
                if l <= 2:
                    da.append(((0,l,t), np.transpose(data).iloc[l,t]),)
                else:
                    da.append(((1,(l-3),t), np.transpose(data).iloc[l,t]),)                
        Y = tupledict(da)
        print(Y)

        #L = tupledict({(0,0):1, (0,1): 1, (0,2): 1, (0,3): 0, (0,4): 0, (0,5): 0})
        #N=T*nL
        #print('T is ' + str(T)+', Groups number is ' + str(nL))

        # Set hidden_state_dim(N0),Variables_num(L), Sample(S), Time(T)
        N0 = 2        
        L = 2
        S = 3
        T = 8

        # Create variables[Variables_num, Sample, Time][2,3,8]
        # L = m.addVars(1, nL, name="L", vtype='B')
        F = m.addVars(N0,N0,(T+1), name="F",vtype='C')
        X = m.addVars(N0,S,(T+1), name="X", vtype='C')
        Sigma = m.addVars(N0,L,(T+1), name="Sigma", vtype='C')
        G = m.addVars(L,N0,(T+1), name="G",vtype='C')      
        V = m.addVars(L,S,(T+1), name="V", vtype='C') 
        K = m.addVars(N0,L,(T+1), name="K",vtype='C')
        P = m.addVars(N0,N0,(T+1), name="P",vtype='C')
        
        # Dummy variables
        li = m.addVars(L,S,T, name="li", vtype='C')
        Res = m.addVars(L,S,T, name="Res", vtype='C')
        ResS = m.addVars(L,L,T, name="ResS", vtype='C')
        w = m.addVars(N0,L,(T+1), name="w", vtype='C')
        ww = m.addVars(N0,L,(T+1), name="ww", vtype='C')
        KV = m.addVars(N0,L,(T+1), name="KV", vtype='C')
        PG = m.addVars(N0,L,T, name="PG", vtype='C')
        GPG = m.addVars(L,L,T, name="GPG", vtype='C')
        InvGPG = m.addVars(L,L,T, name="InvGPG", vtype='C')
        FPG = m.addVars(N0,L,T, name="FPG", vtype='C')
        
        # Piecewise-linear variables
        PWL = m.addVars(L,L,T, name="PWL", vtype='C')
        yy = m.addVars(L,L,T, name="yy", vtype='C')
        xs = [0.01*i for i in range(32)]
        ys = [math.log(p) if p != 0 else 0 for p in xs]  
        #Inv_ = np.tile(np.identity(L), (T,1)).reshape(T,L,L)
        #Inv = []
        #for t in range(T):
        #    for l in range(L):
        #        for ll in range(L):
        #            Inv.append(((l,ll, t), Inv_[t,ll,l]),)
        #Inv = tupledict(Inv)
        ############# Initiations
        
        
        #model.addVars(2, 3)
        #model.addVars([0, 1, 2], ['m0', 'm1', 'm2'])
        #print("This model has",n0*n0* 2+n0*nL +n0*(T+1)* 2+n0*T* 2+T*nL* 3+n1*n1* 2+n1*nL +n1*(T+1)* 2 +n1*T* 2,"decision variables.")
        #InvGPG_ = pd.DataFrame(InvGPG_)
        #InvGPG = np.linalg.inv(np.array(InvGPG_))
        #InvGPG = tuple(InvGPG)
        # X = tupledict(X) 
        # obj += gp.quicksum(ys)      
 
        obj = gp.quicksum(ResS[l,ll,t]*InvGPG[l,ll,t] for l in range(L) for ll in range(L) for t in range(T))
        obj += yy.sum()
        
        #m.addGenConstrPWL(x_, yy, xs, ys, 'pwl')


        m.setObjective(obj, GRB.MINIMIZE)


        # AddConstrs
        ## Res_Square
        m.addConstrs(((li[l,s,t] == G[l,n_,t]*X[n_,s,t]) for l in range(L) for n_ in range(N0) for t in range(T) for s in range(S)),name="li")   
        m.addConstrs(((Res[l,s,t] == Y[l,s,t]- li[l,s,t]) for l in range(L) for t in range(T) for s in range(S)),name="Res")
        m.addConstrs(((ResS[l,l,t] == Res[l,s,t]*Res[l,s,t]) for l in range(L) for t in range(T) for s in range(S)),name="ResS")
        
        ## P
        m.addConstrs(((w[n_,n_,t+1] == F[n_,n_,t+1]- K[n_,l,t+1]*G[l,n_,t+1]) for l in range(L) for n_ in range(N0) for t in range(T)),name="w")   
        m.addConstrs(((ww[n_,n_,t+1] == w[n_,n_,t+1]*P[n_,n_,t]) for l in range(L) for n_ in range(N0) for t in range(T)),name="ww")   
        m.addConstrs(((KV[n_,ll,t+1] == K[n_,l,t+1]*V[l,ll,t+1]) for l in range(L) for ll in range(L) for n_ in range(N0) for t in range(T)),name="KV")
        m.addConstrs(((P[n_,n_,t+1] == ww[n_,l,t+1]*w[n_,l,t+1]+Sigma[n_,l,t]+ KV[n_,l,t+1]*K[n_,l,t+1]) for l in range(L) for n_ in range(N0) for t in range(T)),name="P")   
        
        ## K
        # G[l,n_,t] = GT[n_,l,t]
        m.addConstrs(((PG[n_,l,t] == P[n_,n_,t]*G[l,n_,t]) for l in range(L) for n_ in range(N0) for t in range(T)),name="PG")   
        m.addConstrs(((GPG[l,l,t] == G[l,n_,t]*PG[n_,l,t]) for l in range(L) for n_ in range(N0) for t in range(T)),name="GPG")   
        ## (GPG+V)^(-1)
        # m.addConstrs((Inv[l,ll,t] == InvGPG[l,ll,t]*(GPG[l,ll,t]+V[l,ll,t])) for l in range(L) for ll in range(L) for t in range(T))   
        m.addConstrs(((FPG[n_,l,t] == F[n_,n_,t]*PG[n_,l,t]) for l in range(L) for n_ in range(N0) for t in range(T)),name="FPG")   
        m.addConstrs(((PWL[l,ll,t] == GPG[l,ll,t]+V[l,ll,t]) for l in range(L) for ll in range(L) for t in range(T)),name="PWL_X")   
        
        # Add piecewise-linear constraint
        for t in range(T):
            for i in range(L):
                for j in range(L):
                    m.addGenConstrPWL(PWL[i, j, t], yy[i, j, t], xs, ys, 'pwl') 
        m.addConstrs(((K[n_,l,t]*(GPG[l,ll,t]+V[l,ll,t]) == FPG[n_,ll,t]) for l in range(L) for ll in range(L) for n_ in range(N0) for t in range(T)),name="K1")   
        m.addConstrs(((K[n_,ll,t] == FPG[n_,l,t]*InvGPG[l,ll,t]) for l in range(L) for ll in range(L) for n_ in range(N0) for t in range(T)),name="K2")   
        
        ## X
        m.addConstrs(((X[n_,s,(t+1)] == F[n_,n_,t]*X[n_,s,t]+ K[n_,l,t]*Res[l,s,t]) for l in range(L) for n_ in range(N0) for t in range(T) for s in range(S)),name="X")   
        
        m.update()

        # Solve it!
        m.Params.NonConvex = 2
        #m.setParam('OutputFlag', 0)

        
        m.optimize()
        
        
        if m.status == GRB.Status.OPTIMAL:
            print(f"THIS IS OPTIMAL SOLUTION")
            print(f"Optimal objective value: {m.objVal}")
        else:
            m.computeIIS()
            m.write("model.ilp")
            
            
        # Print solution
        print(f"Optimal values:")
        
        data_dict = {'K': [m.getAttr("x",K)[h] for h in m.getAttr("x",K)],
                     'X': [m.getAttr("x",X)[h] for h in m.getAttr("x",X)],
                     'V': [m.getAttr("x",V)[h] for h in m.getAttr("x",V)],
                     'G': [m.getAttr("x",G)[h] for h in m.getAttr("x",G)]}
        df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        
        '''def X_Pred(row):
               return Fdash.X*row["phi"]+row["q"]

        df["X_Pred"]=df.apply(lambda row:X_Pred(row),axis=1)'''
        print(df)
        return df



import gurobipy as gp
from gurobipy import *
import pandas as pd
import numpy as np

Rawdata = pd.read_csv('HeartRate_MIT_Test.csv',sep=',',header='infer',nrows=8)
Y = pd.DataFrame(Rawdata.drop('Time',axis=1))
#print(Y)
# k cluster no.

y_pred = LogL_Gurobi().estimate(Y,K=2)


