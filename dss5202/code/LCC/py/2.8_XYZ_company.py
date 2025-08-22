# -*- coding: utf-8 -*-
# XYZ_company.py
""" XYZ Company Problem """
import numpy_financial as npf

I = -12000
A = 5310 - 3000
SV = 2000
N = 5
marr = 0.1

# Using pv function to compute NPV
NPV = I - npf.pv(marr, N, A, SV)
print(f"NPV = {NPV:,.2f}")

# Using npv function to compute NPV
CF = [I, A, A, A, A, A+SV]
NPV = npf.npv(marr, CF)
print(f"Cash flows = {CF}")
print(f"NPV = {NPV:,.2f}")

# Using rate function to compute IRR
IRR = npf.rate(N, A, I, SV, guess=0.1)
print(f"IRR = {IRR:.6f}")

# Using irr function to compute IRR
IRR = npf.irr(CF)
print(f"IRR = {IRR:.6f}")

