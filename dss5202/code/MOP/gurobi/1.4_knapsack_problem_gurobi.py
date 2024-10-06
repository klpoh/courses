# -*- coding: utf-8 -*-
# 1.4_knapsack_problem_gurobi.py
import gurobipy as gp
from gurobipy import GRB
""" Knapsack Problem Gurobi (2024 09 22) """

# Sets or Indices
Items = ['I1','I2','I3','I4','I5', 'I6', 'I7', 'I8']

# Data
Volumes = [1, 3, 4, 3, 3, 1, 5, 10 ]
Values =  [2, 9, 3, 8, 10, 6, 4, 10 ]
Cap = 15

## Parameters.
vol = {i: vo  for i, vo  in zip(Items, Volumes)}
val = {i: va  for i, va  in zip(Items, Values)}

## Model
m = gp.Model("KP_problem")

## Decision Variables
# x[i] = 1 if item  i is included, = 0 otherwise
x = m.addVars(Items, vtype=GRB.BINARY, name="x")

# Objective function
# Minimize the total value
m.setObjective(
    gp.quicksum(val[i]*x[i] for i in Items), GRB.MAXIMIZE)
    
## Constraints
# Supplies from each plant p are limited by its capacity
c1 = m.addConstr(
    gp.quicksum(vol[i]*x[i] for i in Items) <= Cap, name='Capacity')

# Save model for inspection/debugging
# m.write('KP.lp')

# Solve the model
m.optimize()

# Print optimal solutions if found
if m.status == GRB.Status.OPTIMAL:
    print("\nOptimal Solution:")
    print(f"Obj Value = {m.objVal}")

    for i in Items:
        if x[i].x == 1:
            print(f"Pack item {i}")
