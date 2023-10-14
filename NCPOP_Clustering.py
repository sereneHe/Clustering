class NCPOLR(object):
    """Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py
    
    Examples
    >>> L = [1,1,0,0,0] 
    >>> level = 1
    >>> y_pred = NCPOLR().estimate(HBP_Data,L,level)
    >>> print(y_pred)
    """
    
    def __init__(self, **kwargs):
        super(NCPOLR, self).__init__()
        
    def estimate(self, X, L, level):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        X: array
            Variable seen as input
        L: array
            Variable seen as Clustering

        Returns
        -------
        y_predict: array
            regression predict values of y or residuals
        obj: num
            Objective value in optima
        """
        
        T = len(X)
        N = len(X.iloc[0,:])
        if(N != len(L)):
            print('Wrong dimension between data and clustering!')
    
        #Two groups
        X_Group0 = X.iloc[:,np.flatnonzero(L)]
        X_Group0 = np.array(X_Group0).reshape(-1)
        N0 = len(X_Group0)
        
        X_Group1 = X.iloc[:,np.flatnonzero(list(abs(np.asarray(L) -1)))]
        X_Group1 = np.array(X_Group1).reshape(-1)
        N1 = len(X_Group1)

    
        # Decision Variables
        G0 = generate_operators("G0", n_vars=1, hermitian=True, commutative=False)[0]
        Fdash0 = generate_operators("Fdash0", n_vars=1, hermitian=True, commutative=False)[0]
        m0 = generate_operators("m0", n_vars=N0+1, hermitian=True, commutative=False)
        q0 = generate_operators("q0", n_vars=N0, hermitian=True, commutative=False)
        p0 = generate_operators("p0", n_vars=N0, hermitian=True, commutative=False)
        f0 = generate_operators("f0", n_vars=N0, hermitian=True, commutative=False)
        G1 = generate_operators("G1", n_vars=1, hermitian=True, commutative=False)[0]
        Fdash1 = generate_operators("Fdash1", n_vars=1, hermitian=True, commutative=False)[0]
        m1 = generate_operators("m1", n_vars=N1+1, hermitian=True, commutative=False)
        q1 = generate_operators("q1", n_vars=N1, hermitian=True, commutative=False)
        p1 = generate_operators("p1", n_vars=N1, hermitian=True, commutative=False)
        f1 = generate_operators("f1", n_vars=N1, hermitian=True, commutative=False)
        #L = generate_operators("L", n_vars=N, hermitian=True, commutative=False)

        # Objective
        obj = sum(( X_Group0[i]-f0[i])**2 for i in range(N0)) + 0.0005*sum(p0[i]**2 for i in range(N0)) + 0.0001*sum(q0[i]**2 for i in range(N0))+sum(( X_Group1[i]-f1[i])**2 for i in range(N1)) + 0.0005*sum(p1[i]**2 for i in range(N1)) + 0.0001*sum(q1[i]**2 for i in range(N1))
    
        # baseline
        #dm = distance.cdist(X,f, 'euclidean')
        #dm = distance.cdist(X,f, lambda u, v: np.sqrt(((u-v)**2).sum()))
        #obj = distance.cdist(X_Group0,f0, 'euclidean').sum()+distance.cdist(X_Group1,f1, 'euclidean').sum()
        #print(X_Group0.iloc[t,n],X_Group1.iloc[t,m])
        #obj = sum((X_Group0.iloc[t,n]-f0[t])**2 for n in range(N0) for t in range(T)) + 0.0005*sum(p0[t]**2 for t in range(T)) + 0.0001*sum(q0[t]**2 for t in range(T)) + sum((X_Group1.iloc[t,m]-f1[t])**2 for m in range (N1) for t in range(T)) + 0.0005*sum(p1[t]**2 for t in range(T)) + 0.0001*sum(q1[t]**2 for t in range(T))
        
        # DTW
        #alignment0 = dtw(X_Group0,f0, keep_internals=True)
        #alignment1 = dtw(X_Group1,f1, keep_internals=True)
        #obj = alignment0.distance+alignment1.distance
    
        # Constraints
        ine1 = [f0[t] - Fdash0*m0[t+1] - p0[t] for t in range(N0)]
        ine2 = [-f0[t] + Fdash0*m0[t+1] + p0[t] for t in range(N0)]
        ine3 = [m0[t+1] - G0*m0[t] - q0[t] for t in range(N0)]
        ine4 = [-m0[t+1] + G0*m0[t] + q0[t] for t in range(N0)]
        ine5 = [f1[t] - Fdash1*m1[t+1] - p1[t] for t in range(N1)]
        ine6 = [-f1[t] + Fdash1*m1[t+1] + p1[t] for t in range(N1)]
        ine7 = [m1[t+1] - G1*m1[t] - q1[t] for t in range(N1)]
        ine8 = [-m1[t+1] + G1*m1[t] + q1[t] for t in range(N1)]
        ines = ine1+ine2+ine3+ine4+ine5+ine6+ine7+ine8

        # Solve the NCPO
        sdp = SdpRelaxation(variables = flatten([G0,Fdash0,f0,p0,m0,q0,G1,Fdash1,f1,p1,m1,q1]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')
        '''
        with sdp.SolverFactory("mosek") as solver:
            # options - MOSEK parameters dictionary, using strings as keys (optional)
            # tee - write log output if True (optional)
            # soltype - accepts three values : bas, itr and itg for basic,
            # interior point and integer solution, respectively. (optional)
            solver.solve(model, options = {'dparam.optimizer_max_time':  100.0, 
                                           'iparam.intpnt_solve_form':   int(mosek.solveform.dual)},
                                tee = True, soltype='itr')

            # Save data to file (after solve())
            solver._solver_model.writedata("dump.task.gz")
            '''
        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual, sdp.status)

        if(sdp.status != 'infeasible'):
            print('ok.')
            f0_pred = []
            p0_pred = []
            m0_pred = []
            q0_pred = []
            f1_pred = []
            p1_pred = []
            m1_pred = []
            q1_pred = []
            for t in range(N0):
                f0_pred.append(sdp[f0[t]])
                #p0_pred.append(sdp[p0[t]])
                #m0_pred.append(sdp[m0[t]])
                #q0_pred.append(sdp[q0[t]])
            for t in range(N1):
                f1_pred.append(sdp[f1[t]])
                #p1_pred.append(sdp[p1[t]])
                #m1_pred.append(sdp[m1[t]])
                #q1_pred.append(sdp[q1[t]])
            #print(f0_pred,p0_pred,m0_pred,q0_pred)
            #print(f1_pred,p1_pred,m1_pred,q1_pred)
            Para = [sdp[G0],sdp[Fdash0],sdp[G1],sdp[Fdash1]]
            return f0_pred,f1_pred, Para

        else:
            print('Cannot find feasible solution.')
            return

        

        # nrmse_sim = 1-sqrt(sdp[sum((K[i]-f[i]+q[i])**2 for i in range(T))])/sqrt(sum((K[i]-np.mean(K))**2 for i in range(T)))

        '''if(sdp.status != 'infeasible'):
            f0_pred = []
            f1_pred = []
            params1 = sdp[G0]
            params2 = sdp[Fdash0]
            params3 = sdp[G1]
            params4 = sdp[Fdash1]

            for t in range(T):
                f0_pred.append(sdp[f0[t]])
                f1_pred.append(sdp[f1[t]])
            params = [params1, params2, params3, params4]
            return f0_pred, f1_pred, params,sdp
        else:
            print('Cannot find feasible solution.')'''
        
        
