# -*- coding: utf-8 -*-
# 1.2_work_schedule_winston_gurobi.py
import gurobipy as gp
from gurobipy import GRB
""" Winston's Weekly Workforce Planning Problem (2024 10 06) """

# Indices
Days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# Data
Workers_req = [17, 13, 15, 19, 14, 16, 11]

# Covers[i][j] = 1 if a staff who starts on day j is also working on day i 
Covers = [[1, 0, 0, 1, 1, 1, 1],
          [1, 1, 0, 0, 1, 1, 1],
          [1, 1, 1, 0, 0, 1, 1],
          [1, 1, 1, 1, 0, 0, 1],
          [1, 1, 1, 1, 1, 0, 0],
          [0, 1, 1, 1, 1, 1, 0],
          [0, 0, 1, 1, 1, 1, 1]]

# Parameters
b = { d : req for  d, req in zip(Days, Workers_req) }

A = {(d1, d2): Covers[i][j] 
         for i, d1 in enumerate(Days)
             for j, d2 in enumerate(Days) }

## Model
m = gp.Model("Workforce planning")

## Decision Variables
# x[i] = number of workers starting work on Day i.
x = m.addVars(Days, vtype=GRB.INTEGER, lb = 0, name="x")

# Objective function
# Minimize the total number of workers employed
m.setObjective(
    gp.quicksum(x[j] for j in Days), GRB.MINIMIZE )
    
## Constraints
# The required number of workers for each day must be satisfied
m.addConstrs(
    (gp.quicksum(A[i,j]*x[j] for j in Days) >= b[i] 
             for i in Days), name = 'Covers')

# Save model for inspection/debugging
# m.write('Workforce.lp')

# Solve the model
m.optimize()

# Print optimal solutions if found
if m.status == GRB.Status.OPTIMAL:
    print("\nOptimal Solution:")
    print(f"Total number of workers = {m.objVal}")

    for d in Days:
        print(f"Number of workers starting on {d} = {x[d].x:4}")
