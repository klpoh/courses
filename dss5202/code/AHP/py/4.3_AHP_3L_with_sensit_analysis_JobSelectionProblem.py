# -*- coding: utf-8 -*-
""" AHP_3L_with_sensit_analysis_JobSelectionProblem.py """
import numpy as np
from scipy import stats
from scipy.optimize import root
import matplotlib.pyplot as plt

def main():

    """ Solve the 3-Level AHP Job Selection Problem (2024 08 30) """
    
    G = 'Job Satisfaction'
    C = ['Research','Growth','Benefits','Colleages',
         'Location','Reputation']
    AL = ['Company A', 'Company B', 'Company C']
    
    # Pairwise compare criteria w.r.t Goal
    A0 = np.array([[ 1,  1 ,  1,  4,  1,  1/2 ],
                   [ 1,  1,   2,  4,  1,  1/2 ],
                   [ 1, 1/2 , 1,  5,  3,  1/2 ],
                   [1/4,1/4, 1/5, 1, 1/3, 1/3 ],
                   [ 1,   1, 1/3, 3,  1,   1  ],
                   [ 2,   2,  2,  3,  1,   1  ]] )
    
    # Pairwise compare alternatives wrt Research
    A1 = np.array([[ 1, 1/4, 1/2 ],  
                   [ 4,  1,    3 ],
                   [ 2, 1/3,   1 ]])
    
    # Pairwise compare alternatives wrt Growth
    A2 =  np.array([[1, 1/4, 1/5 ], 
                    [4,  1,  1/2 ],
                    [5,  2,   1  ]])
    
    # Pairwise compare alternatives wrt Benefits
    A3 = np.array([[ 1,  3,  1/3 ],
                   [1/3, 1,  1/7 ],
                   [ 3,  7,   1  ]])
    
    # Pairwise compare alternatives wrt Colleages
    A4 = np.array([[ 1,  1/3,  5 ],
                   [ 3,   1,   7 ],
                   [1/5, 1/7,  1 ]])
    
    # Pairwise compare alternatives wrt Location
    A5 = np.array([[ 1,   1,   7 ],
                   [ 1,   1,   7 ],
                   [1/7, 1/7,  1 ]])
    
    # Pairwise compare alternatives wrt Reputation      
    A6 = np.array([[ 1,   7,   9 ],
                   [1/7,  1,   2 ],
                   [1/9, 1/2,  1 ]])
        
    
    # Compute Criteria weights
    method = 'Algebra'
    u = AHPmat(A0, method=method)
    print("Criteria's Weights")
    for i, cr in enumerate(C):
        print(f"  {cr:10}: {u[i]:.6f}")
        
    # Compute alternatives' weights wrt each criterion
    W = np.array([ AHPmat(A, method=method) 
                      for A in [A1, A2, A3, A4, A5, A6] ]).T
    print("\nAlternatives' local Weights")
    print(W)
    
    # Compute alternative's global weights
    wG = np.dot(W, u)
    print(f"\nAlternatives' global weight wrt {G}")
    for i, coy in enumerate(AL):
        print(f"  {coy}: {wG[i]:.6f}")

    # Perform sensitivity analysis on Criteria
    for k, cr in enumerate(C):
        WG_dict = {}
        for p in np.linspace(0,1,11):
            adj_u = renorm_wt(p, k, u)
            WG_dict[p] = np.dot(W, adj_u)
        rainbow_diagram(WG_dict, AL, cr, base_val=u[k])
        
        
def rainbow_diagram(w_dict, alternatives, criterion, base_val=None):
    """ Plot the rainbow diagram 
    Parameters:
      w = dictionary of array of alternative weights of the form
            { p : [w1, w2, ..., wn ]  where 0 <= p <= 1.
      alternatives = list of alternatives
      criterion = criterion name
      base_val = base value of criterion being varied
    """
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(w_dict.keys(), w_dict.values(), label=alternatives, lw='2')
    if base_val is not None:
        ax.plot([base_val, base_val], [0, 1], '--', color='black')
    ax.set_title(f"Rainbow diagram for criterion {criterion}", 
                 fontsize='x-large')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_yticks(np.linspace(0,1,11))
    ax.set_xlabel(f'Weight of Criterion {criterion}', fontsize='x-large')
    ax.set_ylabel('Weight of Alternatives', fontsize='x-large')
    ax.legend(fontsize='x-large')
    ax.grid()
    plt.show()


def renorm_wt(p, k, base_wt):
    """ Renormalize the weights when one weight is change 
        while keeping all the other weights in their original 
        proportions 
    Parameters:
        p = new value between 0 and 1
        k = index between 0 and n-1
        base_wt = base weights
    Returns:
        a renormalized weight vector 
    """
  
    new_wt = base_wt.copy()
    bal_wt = base_wt.sum() - base_wt[k]
    for i, w in enumerate(base_wt):
        if i != k:
            new_wt[i] = (1-p)*base_wt[i]/bal_wt
    new_wt[k] = p
    return new_wt


def AHPmat(A, method='Power'):
    """ Comput AHP matrix A using chosen method
    Parameter:  A = matrix to evaluate
    Returns:    w, lambda_max, CI, CR
    """
    
    RI=(0.58,0.90,1.12,1.24,1.32,1.41,1.45,1.49,1.51,1.54,1.56,1.57,1.58) 

    def Power(A):
        """ Compute the AHP matrix A using Power Iterations method.
            Parameter:  A = matrix to evaluate
            Returns:    w, lambda_max, CI, CR
        """
        gm = stats.gmean(A, axis=1)   # Use RGM method as initial value
        w = gm/gm.sum()
        max_iter= 1000000
        epsilon = 1.E-12
        for iter in range(max_iter):
            w1 = np.dot(A,w)    # w(k+1) = A w(k) 
            w1 = w1/w1.sum()    # normalize w(k+1)
            if all(np.absolute(w1-w) < epsilon):
                w = w1
                break
            w = w1
        lambda_max = (np.dot(A,w)/w).mean()
        n, _ = A.shape
        CI = (lambda_max-n)/(n-1)
        CR = 0 if n==2 else CI/RI[n-3]
        return w, lambda_max, CI, CR
    
    def Algebra(A):
        """ Compute the AHP matrix A using Power Iterations method.
            Parameter:  A = matrix to evaluate
            Returns:    w, lambda_max, CI, CR
        """            
        n, _ = A.shape
        # Solve for lambda such that Det(A - lambda*I) = 0
        sol = root(lambda x: np.linalg.det(A-np.eye(n)*x), n)
        lambda_max = sol.x[0]
        # Find w by solving a set of linear equations M w = b
        # M = A - lambda_max I for first n-1 rows
        M = A - np.eye(n)*lambda_max  
        # Replace the last row with [1, 1..., 1]
        M[n-1] = np.ones(n)
        b = np.append(np.zeros(n-1), [1])  # b = [0, 0, ..., 1]
        w = np.linalg.solve(M,b)
        CI = (lambda_max-n)/(n-1)
        CR = 0 if n == 2 else CI/RI[n-3]
        return w, lambda_max, CI, CR
    
    
    """ The RGM method is not recommended as you can do better with 
        Algebra or Power method.  You can use it to compare results """
        
    def RGM(A):
        """ Compute the AHP matrix A using the RGM approximation method.
            Parameter:  A = matrix to evaluate
            Returns:    w, lambda_max, CI, CR
        """           
        n, _ = A.shape
        gm = stats.gmean(A, axis=1)   
        w = gm/gm.sum()
        lambda_max = (np.dot(A,w)/w).mean()
        CI = (lambda_max-n)/(n-1)
        CR = 0 if n==2 else CI/RI[n-3]
        return w, lambda_max, CI, CR
    
    # We just need the w vector
    if method=='Power':
        return Power(A)[0]
    elif method=='Algebra':
        return Algebra(A)[0]
    elif method=='RGM':
        return RGM(A)[0]
    else:
        print("Invalid method chosen")
        exit()
 

if __name__=="__main__":
    main()
