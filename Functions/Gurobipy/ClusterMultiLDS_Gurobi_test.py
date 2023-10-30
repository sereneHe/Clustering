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
        e.setParam('TimeLimit', 5*60)
        m = gp.Model(env=e)
        

        # Create indicator variable
        nL = len(np.transpose(X))
        X =[((l,t),np.transpose(X).iloc[l,t]) for l in range(nL) for t in range(T)]
        X = tupledict(X)

        L = tupledict({(0,0):1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1})
        #N=T*nL
        #print('T is ' + str(T)+', Groups number is ' + str(nL))

        # Set hidden_state_dim(n)
        n0 = 2
        n1 = 4

        # Create variables
        #L = m.addVars(1, nL, name="L", vtype='B')
        G0 = m.addVars(n0,n0, name="G0",vtype='C')
        phi0 = m.addVars(n0,(T+1), name="phi0", vtype='C')
        p0 = m.addVars(n0,T, name="p0", vtype='C')
        F0 = m.addVars(nL,n0, name="F0",vtype='C')
        q = m.addVars(nL,T, name="q", vtype='C')        
        f0 = m.addVars(nL,T, name="f0", vtype='C')
        f1 = m.addVars(nL,T, name="f1", vtype='C')
        G1 = m.addVars(n1,n1, name="G1",vtype='C')
        F1 = m.addVars(nL,n1, name="F1",vtype='C')
        phi1 = m.addVars(n1,(T+1), name="phi1", vtype='C')
        p1 = m.addVars(n1,T, name="p1", vtype='C')

        v0 = m.addVars(nL,T, name="v0", vtype='C')
        v1 = m.addVars(nL,T, name="v1", vtype='C')
        w0 = m.addVars(nL,nL, name="w0", vtype='C')
        w1 = m.addVars(nL,nL, name="w1", vtype='C')
        u = m.addVars(nL,nL, name="u", vtype='C')

        #model.addVars(2, 3)
        #model.addVars([0, 1, 2], ['m0', 'm1', 'm2'])
        print("This model has",n0*n0* 2+n0*nL +n0*(T+1)* 2+n0*T* 2+T*nL* 3+n1*n1* 2+n1*nL +n1*(T+1)* 2 +n1*T* 2,"decision variables.")

        obj = gp.quicksum(w0[l,l]*L[0,l]+w1[l,l]*(1-L[0,l]) for l in range(nL) for t in range(T) )
        obj += gp.quicksum(0.0005*p0[n_,t]*p0[n_,t] for n_ in range(n0) for t in range(T)) 
        obj += gp.quicksum(0.0001*L[0,l]*u[l,l] for l in range(nL) for t in range(T)) 
        #obj += gp.quicksum(w1[l,l]*(1-L[0,l]) for l in range(nL) for t in range(T)) 
        obj += gp.quicksum(0.0005*p1[n_,t]*p1[n_,t] for n_ in range(n1) for t in range(T)) 
        obj += gp.quicksum(0.0001*(1-L[0,l])*u[l,l] for l in range(nL) for t in range(T)) 


        m.setObjective(obj, GRB.MINIMIZE)

        # AddConstrs
        m.addConstrs((v0[l,t] == F0[l,n_]*phi0[n_,(t+1)]) for l in range(nL) for n_ in range(n0) for t in range(T))   
        m.addConstrs((v1[l,t] == F1[l,n_]*phi1[n_,(t+1)]) for l in range(nL) for n_ in range(n1) for t in range(T)) 
        m.addConstrs((w0[l,l] == (X[l,t]-f0[l,t])*(X[l,t]-f0[l,t])) for l in range(nL) for t in range(T))
        m.addConstrs((w1[l,l] == (X[l,t]-f1[l,t])*(X[l,t]-f1[l,t])) for l in range(nL) for t in range(T))
        m.addConstrs((u[l,l] == q[l,t]*q[l,t]) for l in range(nL) for t in range(T))
        m.addConstrs((phi0[n_,(t+1)] == G0[n_,n_]*phi0[n_,t] + p0[n_,t]) for n_ in range(n0) for t in range(T))  
        m.addConstrs((L[0,l] *f0[l,t] == L[0,l] * v0[l,t] + L[0,l] * q[l,t]) for l in range(nL) for t in range(T))  
        m.addConstrs((phi1[n_,(t+1)] == G1[n_,n_]*phi1[n_,t] + p1[n_,t]) for n_ in range(n1) for t in range(T))  
        m.addConstrs(((1-L[0,l])*f1[l,t] == (1-L[0,l])*v1[l,t] + (1-L[0,l])*q[l,t]) for l in range(nL) for t in range(T))  
        m.update()

        # Solve it!
        m.Params.NonConvex = 2
        #m.setParam('OutputFlag', 0)

        
        m.optimize()
        if m.objVal <= 1:
            print(f"Optimal objective value: {m.objVal}")

            print(f"m.status is " + str(m.status))
            print(f"GRB.OPTIMAL is "+ str(GRB.OPTIMAL))

            if m.status == GRB.Status.OPTIMAL:
                print(f"THIS IS OPTIMAL SOLUTION")
            else:
                print(f"THIS IS NOT OPTIMAL SOLUTION")

        '''
        if m.status == GRB.Status.OPTIMAL:
            print(f"THIS IS OPTIMAL SOLUTION")
            print(f"Optimal objective value: {m.objVal}")
            
            # Print solution
            print(f"Optimal L value:")
            print(np.array([m.getAttr("x",L)[h] for h in m.getAttr("x",L)]))  
        
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


