# -*- coding: utf-8 -*-
import pulp
""" Traveling Scaleman Problem 5 nodes PuLP (2024 10 12) """
# Sets or Indices
Nodes = ['S1', 'S2', 'S3', 'S4', 'S5']
Links = [(x,y) for x in Nodes for y in Nodes if x!=y]

# Data
Dist_data = [[ 0,  132, 217, 164,  58],
             [132,  0,  290, 201,  79],
             [217, 290,  0,  113, 303],
             [164, 201, 113,  0,  196],
             [ 58,  79, 303, 196,  0 ]]
    
# parameters dictionaries
dist = {(x,y) : Dist_data[i][j] for i, x in enumerate(Nodes) 
        for j, y in enumerate(Nodes) if i != j}
       
## Model
m = pulp.LpProblem("TSP", pulp.LpMinimize)

## Decision variables
x = {(i,j): pulp.LpVariable(f'travel({i},{j})', cat=pulp.LpBinary) 
         for i, j in Links}

# Subtours breaking vars
u = {i: pulp.LpVariable(f'u({i})', lowBound=0, upBound=None, 
        cat=pulp.LpContinuous) for i in Nodes[1:]}

## Objective function
# Objective function 
m += pulp.lpSum(x[i,j]*dist[i,j] for i,j in Links), 'Total_Distance'
 
# Constraints
# Can only leaves one node
for i in Nodes:
    m += pulp.lpSum(x[i,j] for j in Nodes if j!=i) == 1, f'Out({i})'
# Can only enter one node
for j in Nodes:
    m += pulp.lpSum(x[i,j] for i in Nodes if i!=j) == 1, f'In({j})'
# Subtours breaking constraints
N = len(Nodes)
for i in Nodes[1:]:
    for j in Nodes[1:]:
        if i != j:
            m += u[i]-u[j] + N*x[i,j] <= N-1, f'Break({i},{j})'
        
# Write the model to an .lp file for inspection
# m.writeLP("TSP.lp")

# Listof available solvers: pulp.listSolvers(True)
# solver = pulp.GUROBI(msg=False)
# solver = pulp.GUROBI_CMD(msg=False)
solver = pulp.GLPK_CMD(msg=True)
# solver = pulp.CPLEX_CMD(msg=True)
# solver = pulp.CPLEX_PY(msg=False)
# solver = pulp.PULP_CBC_CMD(msg=False)

# Solve the model 
m.solve(solver)

print(f"Status: {pulp.LpStatus[m.status]}")
print(f"Objective value = {pulp.value(m.objective):,.2f}")

if pulp.LpStatus[m.status]=='Optimal':
    
    # Trace Optimal Tour
    next_node = {i : j for i,j in Links if x[i,j].varValue==1 }
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
