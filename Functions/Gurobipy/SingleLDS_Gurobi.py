class SingleLDS_Gurobi(object):
    """Clustering Estimator based on Gurobi
    """
    
    def __init__(self, **kwargs):
        super(SingleLDS_Gurobi, self).__init__()
        
    def load_heartrate(as_series=False):
        rslt = np.array([
            84.2697, 84.2697, 84.0619, 85.6542, 87.2093, 87.1246,
            86.8726, 86.7052, 87.5899, 89.1475, 89.8204, 89.8204,
            90.4375, 91.7605, 93.1081, 94.3291, 95.8003, 97.5119,
            98.7457, 98.904, 98.3437, 98.3075, 98.8313, 99.0789,
            98.8157, 98.2998, 97.7311, 97.6471, 97.7922])

        if as_series:
            return pd.Series(rslt)
        return rslt
    
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


        Examples
        -------
        >>> import gurobipy as gp
        >>> from gurobipy import GRB
        >>> import pandas as pd
        >>> import numpy as np
        >>> import sys
        
        >>> test_data = SingleLDS_Gurobi().load_heartrate()
        >>> #print(test_data)
        >>> y_pred = SingleLDS_Gurobi().estimate(test_data,K=2)
        
        """

        #X = list(np.array(X).reshape(-1))
        
        # Create a new model
        e = gp.Env()
        e.setParam('TimeLimit', 1*10)
        m = gp.Model(env=e)
        N = len(X)
        L = 4
        print('N,X')

        # Create variables
        G = m.addVar(vtype='C', name="G")
        Fdash = m.addVar(vtype='C', name="Fdash")
        phi = m.addVars((N+1), name="phi", vtype='C')
        q = m.addVars(N, name="q", vtype='C')
        p = m.addVars(N, name="p", vtype='C')
        f = m.addVars(N, name="f", vtype='C')
        l = m.addVars(L, name="l", vtype='B')
        print("This model has",len(phi)+len(q)+len(p)+len(f)+len(l)+4,"decision variables.")
        m.update() 

        # Set objective function
        obj = gp.quicksum((X[t]-f[t])*(X[t]-f[t]) for t in range(N))  
        obj += gp.quicksum(p[t]*p[t] for t in range(N)) 
        obj += gp.quicksum(q[t]*q[t] for t in range(N)) 

        m.setObjective(obj, GRB.MINIMIZE)

        #AddConstrs
        m.addConstrs((f[t] == Fdash*phi[t+1] + p[t]) for t in range(N))  
        m.addConstrs((phi[t+1] == G*phi[t] + q[t]) for t in range(N))  
        m.update()

        m.Params.NonConvex = 2
        
        #solve
        m.optimize()
        
        m.setParam(GRB.Param.OutputFlag, 0)
        print(f"m.status is " + str(m.status))
        print(f"GRB.Status.OPTIMAL is "+ str(GRB.Status.OPTIMAL))
        if m.status == GRB.Status.OPTIMAL:
            print(f"THIS IS OPTIMAL SOLUTION")
        else:
            print(f"THIS IS NOT OPTIMAL SOLUTION")
        print(f"Optimal objective value: {m.objVal}")
        print(f"Solution values: paras={G.X,Fdash.X}")
        
        
        data_dict = {'phi': [m.getAttr("x",phi)[h] for h in m.getAttr("x",phi)],
                     'p': [m.getAttr("x",p)[h] for h in m.getAttr("x",p)],
                     'q': [m.getAttr("x",q)[h] for h in m.getAttr("x",q)],
                     'f': [m.getAttr("x",f)[h] for h in m.getAttr("x",f)],
                     'X_Pred': [Fdash.X*m.getAttr("x",phi)[h+1]+m.getAttr("x",q)[h] for h in m.getAttr("x",f)]}
        df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        
        '''def X_Pred(row):
               return Fdash.X*row["phi"]+row["q"]

        df["X_Pred"]=df.apply(lambda row:X_Pred(row),axis=1)'''
        df["X"]= X
        f=m.getAttr("x",f)
        print("\nf:")

        print(df)


        return df
    
        #self.printSolution()
        
    def printSolution():
        if m.status == GRB.Status.OPTIMAL:
            print("\nError:%g" % m.objval)
            print("\nF:")
            F=m.getAttr("x",F)
            print("\nG:")
            G=m.getAttr("x",G)
            print("\nphi:")
            phi=m.getAttr("x",phi)
            print("\nq:")
            q=m.getAttr("x",q)
            print("\np:")
            p=m.getAttr("x",p)
        else:
            print("No solution")

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import sys

test_data = SingleLDS_Gurobi().load_heartrate()
#print(test_data)
y_pred = SingleLDS_Gurobi().estimate(test_data,K=2)
