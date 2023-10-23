class SingleLDS_Gurobi(object):
    """Clustering Estimator based on Gurobi
    """
    
    def __init__(self, **kwargs):
        super(SingleLDS, self).__init__()
    
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

        Example
        -------
        >>> import gurobipy as gp
        >>> from gurobipy import GRB
        >>> import pandas as pd
        >>> import numpy as np


        >>> Rawdata = pd.read_csv('HeartRate_MIT_Test.csv',sep=',',header='infer',nrows=6)
        >>> Y = pd.DataFrame(Rawdata.drop('Time',axis=1))
        >>> print(Y)
        >>> # k cluster no.

        >>> test_data = Cluster_Gurobi().load_heartrate()
        >>> y_pred = Cluster_Gurobi().estimate(Y,K=2)

        """

        T = len(X)
        L = len(np.transpose(X))
        X = list(np.array(X).reshape(-1))
        
        # Create a new model
        m = gp.Model()


        # Create variables
        G0 = m.addVar(vtype='C', name="G0")
        G1 = m.addVar(vtype='C', name="G1")
        Fdash0 = m.addVar(vtype='C', name="Fdash0")
        Fdash1 = m.addVar(vtype='C', name="Fdash1")
        #model.addVars(2, 3)
        #model.addVars([0, 1, 2], ['m0', 'm1', 'm2'])
        phi = m.addVars(L*(T+1), name="phi", vtype='C')
        q = m.addVars(L*T, name="q", vtype='C')
        p = m.addVars(L*T, name="p", vtype='C')
        f = m.addVars(L*T, name="f", vtype='C')
        l = m.addVars(L, name="l", vtype='B')
        print("This model has",len(phi)+len(q)+len(p)+len(f)+len(l)+4,"decision variables.")
        #+len(G0)+len(G1)+len(Fdash0)+len(Fdash1)
        m.update() 
        '''
        
        #Create decision variables for the foods to buy
        buy=m.addVars(foods,name="buy")
        #也可以是：
        # buy=[]
        # for f in foods:
        #     buy[f]=m.addVar(name=f)

        #For recycling
        # m.setObjective(sum(buy[f]*cost[f] for f in foods),GRB.MINIMIZE)
        
        x = MODEL.addVars(20, 8, vtype=gurobipy.GRB.BINARY)
        # 1
        for i in range(20):
            MODEL.addConstr(x.sum(i, "*") <= 1)
        # 2
        MODEL.addConstrs(x.sum(i, "*") <= 1 for i in range(20))

        # use '*'  as list[:]
        ((X-f)**2.sum(t, "*"))*(1-l).sum(i, "*")
        
        obj = 0
        # Set objective function
        for t in range(T*L):
            objj = (((X-f)**2).sum(t, "*"))
            obj = obj + objj '''
        
        obj = gp.quicksum((X[t]-f[t])*(X[t]-f[t]) for t in range(L*T)) 
        #obj += gp.quicksum(l[i]*(X[t]-f[t])*(X[t]-f[t]) for t in range(L*T) for i in range(L)) 
        obj += gp.quicksum(0.0005*p[t]*p[t] for t in range(L*T)) 
        obj += gp.quicksum(0.0005*q[t]*q[t] for t in range(L*T)) 

        #sum(( X[t]-f[t])**2* (1-l[i]) + l[i]*(X[t]-f[t])**2) + 0.0005*sum(p[t]**2) + 0.0001*sum(q[t]**2)  
        m.setObjective(obj, GRB.MINIMIZE)

        #m.addConstrs((f[t]* (1-l[i]) + f[t]*l[i] == Fdash0*phi[t+1]* (1-l[i]) + Fdash1*phi[t+1]*l[i] + p[t]*(1-l[i]) + p[t]*l[i]) for t in range(L*T) for i in range(L))  

        #m.addConstrs((phi[t+1]* (1-l[i]) + phi[t+1]* l[i]== G0*phi[t]* (1-l[i])+ G0*phi[t]*i + q[t]* (1-l[i])+ q[t]*l[i]) for t in range(L*T) for i in range(L))  

        #m.update()
        m.addConstrs((f[t] == Fdash0*phi[t+1] + p[t]) for t in range(L*T))  

        m.addConstrs((phi[t+1] == G0*phi[t] + q[t]) for t in range(L*T))  

        m.update()
        
        
        '''
        for i in range(L):
            for t in range(L*T):
                m.addConstr((f[t]* (1-l[i]) + f[t]*l[i] == Fdash0*phi[t+1]* (1-l[i]) + Fdash1*phi[t+1]*l[i] + p[t]*(1-l[i]) + p[t]*l[i]))
                m.addConstr((phi[t+1]* (1-l[i]) + phi[t+1]* l[i])== G0*phi[t]* (1-l[i])+ G0*phi[t]*i + q[t]* (1-l[i])+ q[t]*l[i])
        '''
        # Solve it!
        m.Params.NonConvex = 2
        m.optimize()

        print(f"Optimal objective value: {m.objVal}")

        
        
