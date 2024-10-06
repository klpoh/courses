# -*- coding: utf-8 -*-
# 1.4_facility_loc_set_covering_gurobi.py
import gurobipy as gp
from gurobipy import GRB
""" Facility Location Set Covering (2024 10 06) """

# Sets or Indices
Nodes = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']

# Data
dist_data = [[0, 10, 20, 30, 30, 20],
             [10, 0, 25, 35, 20, 10],
             [20, 25, 0, 15, 30, 20], 
             [30, 35, 15, 0, 15, 25],
             [30, 20, 30, 15, 0, 14],
             [20, 10, 20, 25, 14, 0]]

Cover_dist = 15

# Data dictionaries
# dist[i,j] = distance between node i and node j
dist = {(n1,n2): dist_data[i][j] 
            for i, n1 in enumerate(Nodes)
                for j, n2 in enumerate(Nodes) }

# covered[i,j] = 1 if dist[i,j] <= Cover_dist; = 0 otherwise.
covered = {k: 1 if dist[k] <= Cover_dist else 0 for k in dist.keys() }

# Model
m = gp.Model('Facility_location_set_covering')

# Decision Variables
x = m.addVars(Nodes, vtype=GRB.BINARY, name="x")

# The objective is to minimize the total number of locations selected
m.setObjective(gp.quicksum(x[i] for i in Nodes), GRB.MINIMIZE)

# Constraints
c1 = m.addConstrs(
    (gp.quicksum(covered[i,j]*x[j] for j in Nodes) >= 1 for i in Nodes),
        name = 'Cover')

# Save model
# m.write('Facility_location_set_covering.lp')

# Compute optimal solution
m.optimize()

# Print optimal solutions if found
if m.status == GRB.Status.OPTIMAL:
    print(f"\nNumber of locations selected = {m.objVal}")
    for i in Nodes:
        if x[i].x == 1:
            print(f"  Select location {i}")
