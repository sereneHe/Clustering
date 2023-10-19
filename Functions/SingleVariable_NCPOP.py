class Single_NCPOLR(object):
    """Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py
    
    Examples
    --------
    >>> 
import pandas as pd
import sys
sys.path.append("/home/zhouqua1") 
sys.path.append("/home/zhouqua1/NCPOP") 
from inputlds import*
from functions import*
from ncpol2sdpa import*
import numpy as np

level = 1

m= len(np.transpose(Y))
print(Y)
X_Group0 = Y.iloc[:,np.flatnonzero(L)]
X_Group0 = np.array(X_Group0).reshape(-1)
print(X_Group0)
y_pred = NCPOLR().estimate(X_Group0,level)



    
    """
    
    def __init__(self, **kwargs):
        super(NCPOLR, self).__init__()
        
    def estimate(self, Y, level):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.

        Parameters
        ----------
        Y: array
            Variable seen as effect

        Returns
        -------
        y_predict: array
            regression predict values of y or residuals
        """
        
        T = len(Y)

    
        # Decision Variables
        G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
        Fdash = generate_operators("Fdash", n_vars=1, hermitian=True, commutative=False)[0]
        m = generate_operators("m", n_vars=T+1, hermitian=True, commutative=False)
        q = generate_operators("q", n_vars=T, hermitian=True, commutative=False)
        p = generate_operators("p", n_vars=T, hermitian=True, commutative=False)
        f = generate_operators("f", n_vars=T, hermitian=True, commutative=False)

        # Objective
        obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.0005*sum(p[i]**2 for i in range(T)) + 0.001*sum(q[i]**2 for i in range(T))

        #c1*sum(p[i]**2 for i in range(T)) + c2*sum(q[i]**2 for i in range(T))
    
        # Constraints
        ine1 = [f[i] - Fdash*m[i+1] - p[i] for i in range(T)]
        ine2 = [-f[i] + Fdash*m[i+1] + p[i] for i in range(T)]
        ine3 = [m[i+1] - G*m[i] - q[i] for i in range(T)]
        ine4 = [-m[i+1] + G*m[i] + q[i] for i in range(T)]
        #ine5 = [(Y[i]-f[i])**2 for i in range(T)]
        ines = ine1+ine2+ine3+ine4 #+ine5

        # Solve the NCPO
        sdp = SdpRelaxation(variables = flatten([G,Fdash,f,p,m,q]),verbose = 1)
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
            y_pred = []
            p_pred = []
            m_pred = []
            q_pred = []
            G_pred = []
            Fdash_pred = []
            for i in range(T):
                y_pred.append(sdp[f[i]])
                p_pred.append(sdp[p[i]])
                m_pred.append(sdp[m[i]])
                q_pred.append(sdp[q[i]])
                #print(G,Fdash)
                #G_pred.append(sdp[G])
                #Fdash_pred.append(sdp[Fdash])
            print(y_pred,p_pred,m_pred,q_pred)
            return y_pred,p_pred,m_pred,q_pred
        else:
            print('Cannot find feasible solution.')
            return

