# -*- coding: utf-8 -*-
# 9.2.6_Compute_AHP_matrix_np.linalg.eig_function.py
import numpy as np

""" Compute AHP matrix using np.linalg.eig function """

A = np.array([[1, 1/3, 1/2],
              [3,  1,   3 ],
              [2, 1/3,  1 ]])

eigVal, eigVec = np.linalg.eig(A)
# Get the dominant real eigenvalue and its eigenvector
lambda_max, w = max([(val.real, vec.real) for val, vec in 
                     zip(eigVal, eigVec.T) if np.isreal(val)])
w = w/w.sum()  # Normalize w. Can idealize also.
print(f"lambda_max = {lambda_max:.6f}")
print(f"w = {w}")
n, _ = A.shape 
CI = (lambda_max - n)/(n - 1)
CR = CI/0.58  # RI for size 3 is 0.58
print(f"CI = {CI:.6f}, CR = {CR:.6f}")
