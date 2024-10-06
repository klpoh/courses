# -*- coding: utf-8 -*-
# 1.4_TSP_5_nodes_gurobi.py
import gurobipy as gp
from gurobipy import GRB
""" Traveling Scaleman Problem 5 nodes Gurobi (2024 09 08) """

# Sets
Nodes = ['S1', 'S2', 'S3', 'S4', 'S5']
Links = [(x,y) for x in Nodes for y in Nodes if x!=y]

# Data or Parameters
dist_matrix = [[ 0,  132, 217, 164,  58],
               [132,  0,  290, 201,  79],
               [217, 290,  0,  113, 303],
               [164, 201, 113,  0,  196],
               [ 58,  79, 303, 196,  0 ]]

dist = {(x,y) : dist_matrix[i][j] 
            for i, x in enumerate(Nodes) 
                for j, y in enumerate(Nodes) if i != j}

## Model
m = gp.Model('TSP')

## Decision variables
# x[i,j] = 1 if saleman travels from node i to node j
x = m.addVars(Links, vtype=GRB.BINARY, name="x")

# Subtours breaking vars
u = m.addVars(Nodes[1:], vtype=GRB.CONTINUOUS, name="u")

## Objective function
# Minimize the total distance travelled
m.setObjective(
    gp.quicksum(dist[i,j]*x[i,j] for i,j in Links), GRB.MINIMIZE)

# Constraints
c1 = m.addConstrs(
       (gp.quicksum(x[i,j] for j in Nodes if j!=i) == 1 for i in Nodes), 
       name="flow1")
c2 = m.addConstrs(
       (gp.quicksum(x[i,j] for i in Nodes if i!=j) == 1 for j in Nodes), 
       name="flow2")

# Subtours breaking constraints
n = len(Nodes)
c3 = m.addConstrs(
        (u[i]-u[j]+n*x[i,j] <= n-1 for i in Nodes[1:] for j in Nodes[1:]
             if i != j), name="subtours")

# Save the model in LP format for inspection
# m.write('TSP.lp')

# Solve the model
m.optimize()

if m.status == GRB.Status.OPTIMAL:
    print(f"\nObjective value = {m.ObjVal:.4f}\n")
    # Solutions:
    for k in Links:
        print(f"x{k} = {x[k].x}")
    
    # Trace Optimal Tour
    next_node = {i : j for i,j in Links if x[i,j].x==1 }
    start = Nodes[0]  # Can start at any node.
    tour = [start]
    node = start
    done = False
    print("\nOptimal Tour:")
    while not done:
        print(f"  {node} -> {next_node[node]}")
        node = next_node[node]
        tour = tour+[node]
        done = (node==start)
    print(tour)
