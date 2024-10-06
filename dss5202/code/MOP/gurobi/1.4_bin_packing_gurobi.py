# -*- coding: utf-8 -*-
# 1.4_bin_packing_gurobi.py
import gurobipy as gp
from gurobipy import GRB
""" Bin Packing Problem Gurobi (2024 10 06) """

# Sets
Items = range(1, 10)
Bins = range(1, 10)

# Data
Sizes = [0.5, 0.7, 0.5, 0.2, 0.4, 0.2, 0.5, 0.1, 0.6]

# Parameters
s = { item : size for item, size in zip(Items, Sizes) }

## Model
m = gp.Model('BPP')

## Decision Variables
x = m.addVars(Items, Bins, vtype=GRB.BINARY, name='x')
y = m.addVars(Bins, vtype=GRB.BINARY, name='y')

# Objective function
m.setObjective(
    gp.quicksum(y[j] for j in Bins), GRB.MINIMIZE)
    
## Constraints
m.addConstrs(
    (gp.quicksum(x[i,j]*s[i] for i in Items) <= y[j] 
         for j in Bins), name='Capacity')

m.addConstrs(
    (gp.quicksum(x[i,j] for j in Bins) == 1 
         for i in Items), name='Assign')

# Save model for inspection/debugging
# m.write('BPP.lp')

# Solve the model
m.optimize()

# Print optimal solutions if found
if m.status == GRB.Status.OPTIMAL:
    print("\nOptimal Solution:")
    print(f"Obj Value = {m.objVal} bins")

    k = 0
    for j in Bins:
        if y[j].x == 1:
            k +=1
            print(f"Bin {k}:")
            for i in Items:
                if x[i,j].x == 1:
                    print(f" Item {i} ({s[i]})")
    