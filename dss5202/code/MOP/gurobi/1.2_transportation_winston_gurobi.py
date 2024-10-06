# -*- coding: utf-8 -*-
# 1.2_transportation_winston_gurobi.py
import gurobipy as gp
from gurobipy import GRB
""" Transportation problem Winston (2024 10 06) """

# Sets or Indices
Plants = ['P1','P2','P3']
Cities = ['C1','C2','C3','C4']

## Data. They can be read from an external file
supply_data = [35, 50, 40]
demand_data = [45, 20, 30, 30]
cost_data  = [[8,  6, 10, 9],
               [9, 12, 13, 7],
               [14, 9, 16, 5]]

## Parameters.
num_plants = len(Plants)
num_customers = len(Cities)
# Dictionaries to enable data to be indexed by plants and cities
Supply = { p : s for p, s in zip(Plants, supply_data)}
Demand = { c : d for c, d in zip(Cities, demand_data)}
Cost = {(p, c): cost_data[i][j] 
                  for i, p in enumerate(Plants)
                  for j, c in enumerate(Cities)}

## Model
m = gp.Model("Transportation Problem Winston Example")

## Decision Variables
# x[(i,j)] = quantity shipped from plant i to city j
x = m.addVars(Plants, Cities, name="x")

# Objective function
# Minimize the total cost
m.setObjective(
    gp.quicksum(x[i,j]*Cost[i,j] for i in Plants for j in Cities),
        GRB.MINIMIZE)
    
## Constraints
# Supplies from each plant i are limited by its supply capacity
c1 = m.addConstrs(
    (gp.quicksum(x[i,j] for j in Cities) <= Supply[i] 
         for i in Plants), name = 'Supply')

# Demands at each city j must be met
c2 = m.addConstrs(
    (gp.quicksum(x[i,j] for i in Plants) >= Demand[j]
         for j in Cities), name = 'Demand')

# Save model for inspection/debugging
# m.write('Transportation_problem_winston.lp')

# Solve the model
m.optimize()

# Print optimal solutions if found
if m.status == GRB.Status.OPTIMAL:
    print("\nOptimal Solution:")
    print(f"Total Costs = {m.objVal}")

    for i in Plants:
        for j in Cities:
            if x[i,j].x > 0:
                print(f"Transport {x[i,j].x:2.0f} units from {i} to {j}")
