# -*- coding: utf-8 -*-
# multiple_IRR_example.py
import numpy as np
import numpy_financial as npf
from scipy.optimize import root
import matplotlib.pyplot as plt

# The cash flows
CF = [500, -1000, 0, 250, 250, 250]

# The function whose root we wish to find
NPV = lambda r: npf.npv(r, CF)

# Plot NPV(r) vs r
fig, ax = plt.subplots()
r = np.linspace(0, 1, 101)
ax.plot(r, [NPV(y) for y in r], 'r', lw=2)
ax.set_xlabel("Rate")
ax.set_xticks(np.linspace(0,1,11))
ax.set_ylabel("NPV")
ax.set_xlim(0,1)
ax.grid()
plt.show()

guess=0.2
s1 = root(NPV, x0=guess, options={'xtol': 1E-10})
print(f"guess = {guess},  IRR = {s1.x[0]:.6f}")

guess=0.7
s2 = root(NPV, x0=guess, options={'xtol': 1E-10})
print(f"guess = {guess},  IRR = {s2.x[0]:.6f}")
