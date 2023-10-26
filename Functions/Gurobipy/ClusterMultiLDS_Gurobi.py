class ClusterMultiLDS_Gurobi(object):
    """Clustering Estimator based on Gurobi
    """
    
    def __init__(self, **kwargs):
        super(ClusterMultiLDS_Gurobi, self).__init__()

    def estimate(self, X, K):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        X: array
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
        
        T = len(X)
        
        # Create a new model
        e = gp.Env()
        e.setParam('TimeLimit', 1*10)
        m = gp.Model(env=e)
        
        # Create indicator variable
        nL = len(np.transpose(X))
        #L = m.addVars(1, nL, name="L", vtype='B')
        L = tupledict({(0,0):0, (0,1): 1, (0,2): 0, (0,3): 0, (0,4): 1, (0,5): 1})
        print(L)
        N=T*nL
        print('T is ' + str(T)+', Groups number is ' + str(nL))
        #N0 = L.sum(1, "*")
        #N1 = L.sum(0, "*")
        N0 = 3
        N1 = 3
        #print(N0,N1) 
        
        # Set hidden_state_dim(n)
        n = 2

        # Create variables
        # For model 0
        G0 = m.addVars(n,n, name="G0",vtype='C')
        F0 = m.addVars(N0,n, name="F0",vtype='C')
        phi0 = m.addVars(n,(T+1), name="phi0", vtype='C')
        q0 = m.addVars(N0,T, name="q0", vtype='C')
        p0 = m.addVars(n,T, name="p0", vtype='C')
        f0 = m.addVars(N0,T, name="f0", vtype='C')

        # For model 1
        G1 = m.addVars(n,n, name="G1",vtype='C')
        F1 = m.addVars(N1,n, name="F1",vtype='C')
        phi1 = m.addVars(n,(T+1), name="phi1", vtype='C')
        q1 = m.addVars(N1,T, name="q1", vtype='C')
        p1 = m.addVars(n,T, name="p1", vtype='C')
        f1 = m.addVars(N1,T, name="f1", vtype='C')

        #model.addVars(2, 3)
        #model.addVars([0, 1, 2], ['m0', 'm1', 'm2'])
        print("This model has",T*nL* 2+n*n* 2+n*nL +n*(T+1)* 2+T*nL +n*T* 2,"decision variables.")
        #L = [1,0,1,0,0,0] 
        #X =[((t,l),np.transpose(X).iloc[t,l]) for l in range(nL) for t in range(len(X))]
        #X = np.transpose(X)
        
        
        ###############
        X_Model0_ = L.select()*X 
        X_Model1_ = [(1-i) for i in L.select()]*X
        X = np.transpose(X)
        X = tupledict(X) 
        X_Model0 = pd.DataFrame(X_Model0_).loc[:, (pd.DataFrame(X_Model0_) != 0).any(axis=0)]
        X_Model1 = pd.DataFrame(X_Model1_).loc[:, (pd.DataFrame(X_Model1_) != 0).any(axis=0)]
        N0 = len(np.transpose(X_Model0))
        N1 = len(np.transpose(X_Model1))
        ##################

        #print(X_Model0,X_Model1)
        X_Model0 = np.transpose(X_Model0)
        X_Model1 = np.transpose(X_Model1)
        m.update()
        print(X_Model0,X_Model1)
    
        
        #X = tupledict({X.index:[np.transpose(X)]})
        #X_Model0 = X.prod(L)
        #X_Model00 = X.select(L, 0)
        #X_Model01 = X.select(L, 1)
        
        
        #X_Model0 = [((l,t),L[0,l]*X[l,t]) for l in range(nL) for t in range(T) if L[0,l]==1]
        #X_Model0 = tupledict( X_Model0 ) 
        #X_Model1 = [((l,t),(1-L[0,l])*X[l,t]) for l in range(nL) for t in range(T) if L[0,l]==0]
        #X_Model1 = tupledict( X_Model1 ) 
        #print(L.select(), X_Model0.select(),X_Model1.select())
        
        
        
        #print(X_Model0[0]('Health_1'))
        #print(X.select())
        #print(X.select(0, '*'))
        #print(X.select(1, '*'))
        #print(L.select(1, '*'))
        #X_Model1 = tupledict([(1-L[h]) for h in L]).prod(X)
        #X_Model1 = tupledict([(1-m.getAttr("x",L)[h]) for h in m.getAttr("x",L)]).prod(X)

        

        obj = gp.quicksum((X_Model0.iloc[l,t]-f0[l,t])*(X_Model0.iloc[l,t]-f0[l,t]) for l in range(N0) for t in range(T)) 
        obj += gp.quicksum(0.0005*p0[n_,t]*p0[n_,t] for n_ in range(n) for t in range(T)) 
        obj += gp.quicksum(0.0001*q0[l,t]*q0[l,t] for l in range(N0) for t in range(T)) 
        obj += gp.quicksum((X_Model1.iloc[l,t]-f1[l,t])*(X_Model1.iloc[l,t]-f1[l,t]) for l in range(N1) for t in range(T)) 
        obj += gp.quicksum(0.0005*p1[n_,t]*p1[n_,t] for n_ in range(n) for t in range(T)) 
        obj += gp.quicksum(0.0001*q1[l,t]*q1[l,t] for l in range(N1) for t in range(T)) 


        m.setObjective(obj, GRB.MINIMIZE)

        # AddConstrs
        m.addConstrs((phi0[n_,(t+1)] == G0[n_,n_]*phi0[n_,t] + p0[n_,t]) for n_ in range(n) for t in range(T))  
        m.addConstrs((f0[l,t] == F0[l,n_]*phi0[n_,(t+1)] + q0[l,t]) for l in range(N0) for n_ in range(n) for t in range(T))  
        m.addConstrs((phi1[n_,(t+1)] == G1[n_,n_]*phi1[n_,t] + p1[n_,t]) for n_ in range(n) for t in range(T))  
        m.addConstrs((f1[l,t] == F1[l,n_]*phi1[n_,(t+1)] + q1[l,t]) for l in range(N1) for n_ in range(n) for t in range(T))  
        m.update()

        # Solve it!
        m.Params.NonConvex = 2
        m.optimize()

        print(f"Optimal objective value: {m.objVal}")
                
        print(f"m.status is " + str(m.status))
        print(f"GRB.Status.OPTIMAL is "+ str(GRB.Status.OPTIMAL))
        if m.status == GRB.Status.OPTIMAL:
            print(f"THIS IS OPTIMAL SOLUTION")
        else:
            print(f"THIS IS NOT OPTIMAL SOLUTION")
        print(f"Optimal objective value: {m.objVal}")
        print(f"Solution values: paras={G0.X,Fdash0.X}")
        
        
        data_dict = {'phi0': [m.getAttr("x",phi0)[h] for h in m.getAttr("x",phi0)],
                     'p0': [m.getAttr("x",p0)[h] for h in m.getAttr("x",p0)],
                     'q0': [m.getAttr("x",q0)[h] for h in m.getAttr("x",q0)],
                     'f0': [m.getAttr("x",f0)[h] for h in m.getAttr("x",f0)],
                     'X_Pred0': [Fdash0.X*m.getAttr("x",phi0)[h+1]+m.getAttr("x",q0)[h] for h in m.getAttr("x",f0)],
                    'phi1': [m.getAttr("x",phi1)[h] for h in m.getAttr("x",phi1)],
                     'p1': [m.getAttr("x",p1)[h] for h in m.getAttr("x",p1)],
                     'q1': [m.getAttr("x",q1)[h] for h in m.getAttr("x",q1)],
                     'f1': [m.getAttr("x",f1)[h] for h in m.getAttr("x",f1)],
                     'X_Pred1': [Fdash1.X*m.getAttr("x",phi1)[h+1]+m.getAttr("x",q1)[h] for h in m.getAttr("x",f1)]}
        df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        
        '''def X_Pred(row):
               return Fdash.X*row["phi"]+row["q"]

        df["X_Pred"]=df.apply(lambda row:X_Pred(row),axis=1)'''
        df["X"]= X
        f=m.getAttr("x",f)
        print("\nf:")

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

y_pred = ClusterMultiLDS_Gurobi().estimate(Y,K=2)
