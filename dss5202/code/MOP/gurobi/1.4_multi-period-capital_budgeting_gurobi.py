# -*- coding: utf-8 -*-
# 1.4_multi-period-capital_budgeting_gurobi.py
import gurobipy as gp
from gurobipy import GRB
""" Multi-Period Capital Budgeting Problem (2024 09 08) """

# Sets or Indices
Projects = ['P1','P2','P3','P4','P5']
Years = [1, 2, 3]
Budgets = [25, 25, 25]
Cash_flows = [[5, 1, 8 ],
              [4, 7, 10],
              [3, 9, 2 ],
              [7, 4, 1 ],
              [8, 6, 10]]
Benefits = [20, 40, 20, 15, 30 ]

## Parameters
budget = {y: b for y, b in zip(Years, Budgets)}
cf = {(p, y) : Cash_flows[i][j] 
                for i, p in enumerate(Projects)
                    for j, y in enumerate(Years)}
npv = {p: b for p, b in zip(Projects, Benefits)}

## Model
m = gp.Model("Capital Budgeting")

## Decision Variables
# x[j] = 1 if project j is choose; = 0 otherwise
x = m.addVars(Projects, vtype=GRB.BINARY, name="x")

# Objective function
# Maximize the total NPV
m.setObjective(
    gp.quicksum(x[i]*npv[i] for i in Projects), GRB.MAXIMIZE)

## Constraints
# Budget of each year should not be exceeded
c1 = m.addConstrs(
    (gp.quicksum(x[i]*cf[i,j] for i in Projects) <= budget[j]
         for j in Years), name='Budget')

# Save model for inspection/debugging
# m.write('Capital_budgeting.lp')

# Solve the model
m.optimize()

# Print optimal solutions if found
if m.status == GRB.Status.OPTIMAL:
    print("\nOptimal Solution:")
    print(f"Obj Value = {m.objVal}")
    print("Select:")
    for i in Projects:
        if x[i].x == 1:
            print(f"Project {i}:")
