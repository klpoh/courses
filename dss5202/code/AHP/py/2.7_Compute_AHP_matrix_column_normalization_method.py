# -*- coding: utf-8 -*-
# 9.2.7_Compute_AHP_matrix_column_normalization_method.py
import numpy as np
""" Compute AHP matrix using Column Normalization approximation method """
""" This method is fast but not very accurate. Use RGM method instead """

A = np.array([[1, 1/3, 1/2],
              [3,  1,   3 ],
              [2, 1/3,  1 ]])

# Normalize each column of A, then average across each row.
w = (A/A.sum(axis=0)).mean(axis=1)
print(f"w = {w}")

# Estimate lambda_max using all rows
lambda_max = (np.dot(A,w)/w).mean()
n, _ = A.shape
CI = (lambda_max-n)/(n-1)
CR = CI/0.58  # RI for n=3
print(f"lambda_max= {lambda_max:.6f}, CI = {CI:.6f}, CR= {CR:.6f}")

