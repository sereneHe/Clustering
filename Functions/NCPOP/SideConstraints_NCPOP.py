class Multi_SideInfo_NCPOLR(object):
    """Estimator based on NCPOP Regressor

    References
    ----------
    Quan Zhou https://github.com/Quan-Zhou/Proper-Learning-of-LDS/blob/master/ncpop/functions.py
    Ahmadi, A. A. and El Khadir, B. (2020). Learning dynamical systems with side information. In Learning for Dynamics and Control, pages 718–727. PMLR
    Zhou, Q. and Mareˇcek, J.(2023). Learning of linear dynamical systems as a non-commutative polynomial optimization problem. IEEE Transactions on Automatic Control
    
    Examples
    --------
    >>> from ..Clustering.datasets import load_heartrate
    >>> import pandas as pd
    >>> import sys
    >>> sys.path.append("/home/zhouqua1") 
    >>> sys.path.append("/home/zhouqua1/NCPOP") 
    >>> from inputlds import*
    >>> from functions import*
    >>> from ncpol2sdpa import*
    >>> import numpy as np
    >>> from sympy.physics.quantum import HermitianOperator, Operator    
    >>> Test_Data = load_heartrate().reshape(600,3)
    >>> level = 1
    >>> N = len(np.transpose(Test_Data))
    >>> Multi_SideInfo_NCPOLR().estimate(Test_Data,N,level)
    """
    
    def __init__(self, **kwargs):
        super(Multi_SideInfo_NCPOLR, self).__init__()
    
    
    def generate_multioperators(self, name, n_vars, m_vars, hermitian=None, commutative=False):
        """Generates a number of commutative or noncommutative operators

        :param name: The prefix in the symbolic representation of the noncommuting
                     variables. This will be suffixed by a number from 0 to
                     n_vars-1 if n_vars > 1.
        :type name: str.
        :param n_vars: The number of variables.
        :type n_vars: int.
        :param hermitian: Optional parameter to request Hermitian variables .
        :type hermitian: bool.
        :param commutative: Optional parameter to request commutative variables.
                            Commutative variables are Hermitian by default.
        :type commutative: bool.

        :returns: list of :class:`sympy.physics.quantum.operator.Operator` or
                  :class:`sympy.physics.quantum.operator.HermitianOperator`
                  variables

        :Example:

        >>> generate_multioperators('y', 2, 3, commutative=True)
        #[[y00, y01, y02][y10, y11, y12]]
        """

        variables = []
        variables1 = []
        variables2 = []
        for i in range(n_vars):
            if n_vars > 1:
                var_name1 = '%s%s' % (name, i)
            else:
                var_name1 = '%s' % name
            if hermitian is not None and hermitian:
                variables1.append(HermitianOperator(var_name1))
            else:
                variables1.append(Operator(var_name1))
            variables1[-1].is_commutative = commutative
        for n in range(len(variables1)):        
            for j in range(m_vars):
                if m_vars > 1:
                    var_name = '%s%s' % (variables1[n], j)
                else:
                    var_name = '%s' % variables1[n]
                if hermitian is not None and hermitian:
                    variables2.append(HermitianOperator(var_name))
                else:
                    variables2.append(Operator(var_name))
                variables2[-1].is_commutative = commutative
        var = np.matrix(np.array(variables2).reshape(n_vars,m_vars))
        var = np.array(variables2).reshape(n_vars,m_vars)
        #print(variables1,variables2)
        return var

        
    def estimate(self, Y, m,level):
        """Fit Estimator based on NCPOP Regressor model and predict y or produce residuals.
        The module converts a noncommutative optimization problem provided in SymPy
        format to an SDPA semidefinite programming problem.
        Define a function for solving the NCPO problems with 
        given standard deviations of process noise and observtion noise,
        length of  estimation data and required relaxation level. 

        Parameters
        ----------
        Y: array
            Variable seen as input
        m: int
            dimension of observations
        n: int

        Returns
        -------
        obj: num
            Objective value in optima
        """
        n=7
        T= len(Y)
        Y = np.transpose(Y)

        # Decision Variables
        G = self.generate_multioperators("G", n_vars=n, m_vars=n, hermitian=True, commutative=False)
        Fdash = self.generate_multioperators("Fdash", n_vars=m, m_vars=n, hermitian=True, commutative=False)
        phi = self.generate_multioperators("phi", n_vars=n, m_vars=T+1, hermitian=True, commutative=False)
        q = self.generate_multioperators("q", n_vars=n, m_vars=T, hermitian=True, commutative=False)
        p = self.generate_multioperators("p", n_vars=m, m_vars=T, hermitian=True, commutative=False)
        f = self.generate_multioperators("f", n_vars=m, m_vars=T, hermitian=True, commutative=False)
        

        # Objective
        #obj = sum((Y[i]-f[i])**2 for i in range(T)) + 0.0005*sum(p[i]**2 for i in range(T)) + 0.0001*sum(q[i]**2 for i in range(T))
        obj = sum((Y[mm][i])*2 for i in range(T) for mm in range(m))+ 0.0005*sum(p[mm][i]*2 for i in range(T) for mm in range(m)) + 0.0001*sum(q[nn][i]*2 for i in range(T) for nn in range(n)) 
        
        # Constraints
        ine1 = [f[mm][i] - Fdash[mm][nn]*phi[nn][i+1] - p[mm][i] for nn in range(n) for i in range(T) for mm in range(m)]
        ine2 = [-f[mm][i] + Fdash[mm][nn]*phi[nn][i+1] + p[mm][i] for nn in range(n) for i in range(T) for mm in range(m)]
        ine3 = [phi[nn][i+1] - G[nn][nn]*phi[nn][i] - q[nn][i] for nn in range(n) for i in range(T)]
        ine4 = [-phi[nn][i+1] + G[nn][nn]*phi[nn][i] + q[nn][i] for i in range(T) for nn in range(n)]

        # Add Side Information
        ine5 = [phi[nn][i]-1 for i in range(T) for nn in range(n)] 
        ine6 = [-phi[nn][i]+1 for i in range(T) for nn in range(n)] 
        ines = ine1+ine2+ine3+ine4+ine5+ine6

        # Solve the NCPO
        GG = Operator(np.asarray(G).reshape(-1))
        FFdash = Operator(np.asarray(Fdash).reshape(-1))
        f = Operator(np.asarray(f).reshape(-1))
        q = Operator(np.asarray(q).reshape(-1))
        p = Operator(np.asarray(p).reshape(-1))
        phi = Operator(np.asarray(phi).reshape(-1))
        #print([Operator(GG),Operator(FFdash),Operator(f),Operator(p),Operator(phi),Operator(q)])
    
        sdp = SdpRelaxation(variables = flatten([GG,FFdash,f,p,phi,q]),verbose = 1)
        sdp.get_relaxation(level, objective=obj, inequalities=ines)
        sdp.solve(solver='mosek')

        sdp.write_to_file("slutions.csv")
        sdp.write_to_file('example.dat-s')
        sdp.find_solution_ranks() 
        #sdp.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:/Users/zhouq/Documents/sdpa7-windows/sdpa.exe"})
        print(sdp.primal, sdp.dual,sdp.status)
        return sdp.primal
