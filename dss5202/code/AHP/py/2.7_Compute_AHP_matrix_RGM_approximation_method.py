# -*- coding: utf-8 -*-
# 9.2.7_Compute_AHP_matrix_RGM_approximation_method.py
import numpy as np
from scipy.stats import gmean

""" Compute AHP matrix using Row Geometric Mean approximation method """

A = np.array([[ 1, 1/3, 1/2],
              [ 3,  1,   3 ],
              [ 2, 1/3,  1 ]] )

# Compute the geometric mean of each row, then normalize it.
rgm = gmean(A, axis=1)   
w = rgm/rgm.sum()
print(f"w = {w}")

# Estimate lambda_max using all rows
lambda_max = (np.dot(A,w)/w).mean()
n, _ = A.shape
CI = (lambda_max-n)/(n-1)
CR = CI/0.58
print(f"lambda_max= {lambda_max:.6f}, CI= {CI:.6f}, CR= {CR:.6f}")

