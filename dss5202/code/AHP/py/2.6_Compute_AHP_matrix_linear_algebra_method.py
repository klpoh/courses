# -*- coding: utf-8 -*-
# 9.2.6_Compute_AHP_matrix_linear_algebra_method.py
import numpy as np
from scipy.optimize import root

""" Compute AHP matrix using Linear Algebra method """

A = np.array([[1, 1/3, 1/2],
              [3,  1,   3 ],
              [2, 1/3,  1 ]])

n, _ = A.shape   # Number of elements
I = np.eye(n)    # Identity matrix

# Find lambda_max by finding root of the determinant equation
eq = lambda y: np.linalg.det(A-I*y)
sol = root(eq, x0=n, options={'xtol':1e-12})
lambda_max = sol.x[0]
print(f"lambda_max = {lambda_max:.6f}") 

# Find w by solving a set of linear equations M w = b
M = A - I*lambda_max   # M = A - lambda_max I for first n-1 rows
M[n-1] = np.ones(n)    # Replace the last row with [1, 1..., 1]
b = np.append(np.zeros(n-1), [1])  # b = [0, 0, ..., 1]
w = np.linalg.solve(M,b)
print(f"w = {w}")

# Compute CI and CR
CI = (lambda_max-n)/(n-1)
CR = CI/0.58  # RI = 0.58 for n = 3
print(f"CI= {CI:.6f}, CR= {CR:.6f}")
