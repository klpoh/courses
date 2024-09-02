# -*- coding: utf-8 -*-
# ABC_company.py
""" ABC Company Problem """
import numpy_financial as npf

I = -25000
A = 8000
SV = 5000
N = 5
marr=0.2

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

# Using mirr function to compute MIRR
fin_rate = 0.15
reinvest_rate = 0.20
MIRR = npf.mirr(CF, fin_rate, reinvest_rate)
print(f"MIRR = {MIRR:.6f}")
