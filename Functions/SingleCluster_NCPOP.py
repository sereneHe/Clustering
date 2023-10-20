class Cluster_NCPOLR(object):
    """Clustering Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py
    
    Examples
    --------
    >>> import pandas as pd
    >>> from itertools import product
    >>> import sys
    >>> sys.path.append("/home/zhouqua1") 
    >>> sys.path.append("/home/zhouqua1/NCPOP") 
    >>> from inputlds import*
    >>> from functions import*
    >>> from ncpol2sdpa import*
    >>> import numpy as np

    >>> level = 1
    >>> Rawdata = pd.read_csv('HeartRate_MIT_Test.csv',sep=',',header='infer',nrows=31)
    >>> Y = pd.DataFrame(Rawdata.drop('Time',axis=1))
    >>> print(Y)
    >>> # k cluster no.
    >>> y_pred = Cluster_NCPOLR().estimate(Y,K=2,level)
    """
    
    def __init__(self, **kwargs):
        super(Cluster_NCPOLR, self).__init__()
        
    def estimate(self, X, K, level):
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
        L = len(np.transpose(X))
        X = np.array(X).reshape(-1)
    
        # Decision Variables
        l = generate_operators("l", n_vars=L, hermitian=True, commutative=False)    
        N0 = 2
        '''
        G0 = generate_operators("G0", n_vars=1, hermitian=True, commutative=False)[0]
        Fdash0 = generate_operators("Fdash0", n_vars=1, hermitian=True, commutative=False)[0]
        m0 = generate_operators("m0", n_vars=(N0+1)*T, hermitian=True, commutative=False)
        q0 = generate_operators("q0", n_vars=N0*T, hermitian=True, commutative=False)
        p0 = generate_operators("p0", n_vars=N0*T, hermitian=True, commutative=False)
        G1 = generate_operators("G1", n_vars=1, hermitian=True, commutative=False)[0]
        Fdash1 = generate_operators("Fdash1", n_vars=1, hermitian=True, commutative=False)[0]
        m1 = generate_operators("m1", n_vars=((L-N0)+1)*T, hermitian=True, commutative=False)
        q1 = generate_operators("q1", n_vars=(L-N0)*T, hermitian=True, commutative=False)
        p1 = generate_operators("p1", n_vars=(L-N0)*T, hermitian=True, commutative=False'''

        G0 = generate_operators("G0", n_vars=1, hermitian=True, commutative=False)[0]
        Fdash0 = generate_operators("Fdash0", n_vars=1, hermitian=True, commutative=False)[0]
        G1 = generate_operators("G1", n_vars=1, hermitian=True, commutative=False)[0]
        Fdash1 = generate_operators("Fdash1", n_vars=1, hermitian=True, commutative=False)[0]
        m = generate_operators("m", n_vars= L*(T+1), hermitian=True, commutative=False)
        q = generate_operators("q", n_vars= L*T, hermitian=True, commutative=False)
        p = generate_operators("p", n_vars= L*T, hermitian=True, commutative=False)
        f = generate_operators("f", n_vars= L*T, hermitian=True, commutative=False)


        
        # Objective
        #N0 = T* (L-len(l[l == 0]))
        #N1 = T* len(b)
        
        obj = sum(( X[t]-f[t])**2* (1-l[i]) + l[i]*(X[t]-f[t])**2 for t in range(T*L) for i in range(L)) + 0.0005*sum(p[t]**2 for t in range(L*T)) + 0.0001*sum(q[t]**2 for t in range(L*T))
        #obj = sum((( X_Group0[t]-f0[t])**2*(1-i) +i*(X_Group0[t]-f0[t])**2 + 0.0005*p0[i]**2*(1-i) + 0.0001*q0[i]**2*(1-i) + 0.0005*i*p1[i]**2 + 0.0001*i*q1[i]**2 for t in range(N) for i in range(L)) 
        
        # Constraints
        ine1 = [(f[t]* (1-l[i]) + f[t]*l[i]) - Fdash0*m[t+1]* (1-l[i]) - Fdash1*m[t+1]* l[i] - p[t]* (1-l[i]) - p[t]*l[i] for t in range(L*T)for i in range(L)]
        ine2 = [-(f[t]* (1-l[i]) + f[t]*l[i]) + Fdash0*m[t+1]* (1-l[i]) + Fdash1*m[t+1]* l[i] + p[t]* (1-l[i]) + p[t]*l[i] for t in range(L*T)for i in range(L)]
        ine3 = [(m[t+1]* (1-l[i]) + m[t+1]* l[i])- G0*m[t]* (1-l[i])- G0*m[t]*i - q[t]* (1-l[i])- q[t]*l[i] for t in range(L*T)for i in range(L)]
        ine4 = [-(m[t+1]* (1-l[i]) + m[t+1]* l[i])+ G0*m[t]* (1-l[i])+ G0*m[t]*i + q[t]* (1-l[i])+ q[t]*l[i] for t in range(L*T)for i in range(L)]
        ine5 = [l[i]*(l[i]-1) - 0 for i in range(L)]
        ine6 = [-l[i]*(l[i]-1) + 0 for i in range(L)]
        '''ine7 = [N0-2]
        ine8 = [-N0+2]
        ine9 = [L-N0-2]
        ine10 = [-(L-N0)+2]
        ine7+ine8+ine9+ine10'''
        
        
        ines = ine1+ine2+ine3+ine4+ine5+ine6

        # Solve the NCPO
        sdp = SdpRelaxation(variables = flatten([G0,Fdash0,G1,Fdash1,f,p,m,q,l]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')

        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual, sdp.status)

        if(sdp.status != 'infeasible'):
            f_pred = []
            l_pred = []
            params1 = sdp[G0]
            params2 = sdp[Fdash0]
            params3 = sdp[G1]
            params4 = sdp[Fdash1]

            for t in range(T):
                f_pred.append(sdp[f[t]])
                l_pred.append(sdp[l[t]])
            params = [params1, params2, params3, params4]
            return f_pred, l_pred, params,sdp
        else:
            print('Cannot find feasible solution.')
        
        
