# -*- coding: utf-8 -*-
# 1.4_transportation_problem_with_fixed_facility_costs_gurobi.py
import gurobipy as gp
from gurobipy import GRB
"""Transportation Problem with Fixed Facility Costs (2024 10 06)"""

# Sets 
Facilities = ['F1','F2','F3','F4','F5','F6']
Customers  = ['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10']

## Data. 
# Demand of customers
demands = [50, 68, 80, 50, 60, 79, 66, 72, 66, 63]

# Maximum capacity of facilties
capacities = [170, 200, 180, 240, 190, 210 ]
# Fixed operating cost of facilties
fixed_costs= [1000, 1200, 800, 1300, 900, 1100 ]
# Unit transportation cost from facility to customer
transp_costs = [[4,   7, 14, 14, 13,  7, 4, 12, 7,  7],
                [6,   5,  5, 12, 15, 14, 12, 9, 7, 10],
                [13, 12,  3,  6, 11, 13,  9, 5, 8,  9],
                [10, 13, 11,  7,  3,  8, 11, 9, 12, 8],
                [6,   7,  9, 10, 10,  7,  7, 7, 7,  2],
                [10,  9,  8, 10, 12, 11,  7, 7, 3, 10]]

## Parameters 
# Convert data structure to dictionaries
f = { cost: fixed_costs[i] for i, cost in enumerate(Facilities)}
s = { cap: capacities[i] for i, cap in enumerate(Facilities)}
d = { dem: demands[i] for i, dem in enumerate(Customers)}
c = {(fac, cus): transp_costs[i][j] 
         for i, fac in enumerate(Facilities)
                for j, cus in enumerate(Customers)}

## Model
m = gp.Model("transportation problem with fixed facility costs")

## Decision Variables
# y[i,j] = quantity shipped from facility i to customer j
y = m.addVars(Facilities, Customers, name="y")

# x[i] = 1 if facility i is selected, = 0 otherwise
x = m.addVars(Facilities, vtype=GRB.BINARY, name="x")

# Objective function
# Minimize the total fixed cost + transportation cost
m.setObjective(
    gp.quicksum(f[i]*x[i] for i in Facilities) + \
    gp.quicksum(c[i,j]*y[i,j] for i in Facilities for j in Customers),
        GRB.MINIMIZE)
    
## Constraints
m.addConstrs(
    (gp.quicksum(y[i,j] for i in Facilities) >= d[j] 
         for j in Customers), name = 'Demand')
m.addConstrs(
    (gp.quicksum(y[i,j] for j in Customers) <= s[i]*x[i] 
         for i in Facilities), name = 'Capacity')
   
# Solve the model
# m.Params.OutputFlag = 0
m.optimize()
if m.status == GRB.Status.OPTIMAL:
    print("Optimal Solution:")
    print(f"  Min cost = {m.objVal}")
    for i in Facilities:
        if x[i].x == 1:
         print(f"  Use Facility {i}:")
         for j in Customers:
             if y[i, j].x > 0:
                 print(f"    Ship {y[i,j].x:.2f} to customer {j}") 
