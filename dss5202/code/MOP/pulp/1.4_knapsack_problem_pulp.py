# -*- coding: utf-8 -*-
import pulp
""" Knapsack Problem Pulp (2024 10 12) """

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
m = pulp.LpProblem("knapsack", pulp.LpMaximize)

## Decision Variables
x = { i: pulp.LpVariable(f'x({i})', cat=pulp.LpBinary) for i in Items}

# Objective function
m += pulp.lpSum(val[i]*x[i] for i in Items),'total_value'

## Constraints
m += pulp.lpSum(vol[i]*x[i] for i in Items) <= Cap, 'capacity'

# Save model for inspection/debugging
# m.writeLP('TKP.lp')

# Listof available solvers: pulp.listSolvers(True)
# solver = pulp.GUROBI(msg=True)
# solver = pulp.GUROBI_CMD(msg=False)
# solver = pulp.GLPK_CMD(msg=False)
# solver = pulp.CPLEX_CMD(msg=True)
# solver = pulp.CPLEX_PY(msg=False)
solver = pulp.PULP_CBC_CMD(msg=False)

# Solve the model 
m.solve(solver)
# m.solve()  # default to CBC

if pulp.LpStatus[m.status] == 'Optimal':
    print("\nOptimal Solution:")
    print(f"Obj values = {pulp.value(m.objective)}")

    for i in Items:
        if x[i].varValue == 1:
            print(f"Pack item {i}")
