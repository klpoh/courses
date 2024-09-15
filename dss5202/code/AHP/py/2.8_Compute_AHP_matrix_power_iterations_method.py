# -*- coding: utf-8 -*-
# 9.2.7_Compute_AHP_matrix_power_iterations_method.py
import numpy as np
from scipy.stats import gmean

""" Compute AHP matrix using power iterations method """

A = np.array([[1, 1/3, 1/2],
              [3,  1,   3 ],
              [2, 1/3,  1 ]])

# Initial solution:
#   Use RGM approximation as starting solution.
gm = gmean(A, axis=1)  
w = gm/gm.sum()
#   Use column normalization as initial solution . More iterations needed
# w = (A/A.sum(axis=0)).mean(axis=1) 

# Perform Iterations
max_iter= 1000000
epsilon = 1.E-16
for iter in range(max_iter):
    w1 = np.dot(A,w)    # w(k+1) = A w(k) 
    w1 = w1/w1.sum()    # normalize w(k+1)
    if all(np.absolute(w1-w) < epsilon):
        w = w1
        print(f"Tolerance {epsilon} achieved at iter #{iter}")
        break
    w = w1
print(f"w = {w}")

lambda_max = (np.dot(A,w)/w).mean()
n, _ = A.shape
CI = (lambda_max-n)/(n-1)
CR =  CI/0.58
print(f"lambda_max= {lambda_max:.6f}, CI= {CI:.6f}, CR= {CR:.6f}")
