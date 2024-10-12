# -*- coding: utf-8 -*-
import pulp
""" Facility Location Set Covering PuLP (2024 10 12) """

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
dist = {(n1,n2): dist_data[i][j] for i, n1 in enumerate(Nodes)
            for j, n2 in enumerate(Nodes) }

# covered[i,j] = 1 if dist[i,j] <= Cover_dist; = 0 otherwise.
covered = {k: 1 if dist[k] <= Cover_dist else 0 for k in dist.keys() }

# Model
m = pulp.LpProblem("Facility_location_set_covering", pulp.LpMinimize)

# Decision Variables
x = { k : pulp.LpVariable(f'x_{k}', cat=pulp.LpBinary) for k in Nodes }

# The objective is to minimize the total number of locations selected
m += pulp.lpSum(x[k] for k in Nodes), 'Number_locations_selected'

# Constraints
for i in Nodes:
    m += pulp.lpSum(covered[i,j]*x[j] for j in Nodes) >= 1, f'Cover_{i}'
 
# Save of the model
# m.writeLP('Facility_location_set_covering.lp')

# Listof available solvers: pulp.listSolvers(True)
solver = pulp.GUROBI(msg=False)
# solver = pulp.GUROBI_CMD((msg=False)
# solver = pulp.GLPK_CMD(msg=True)
# solver = pulp.CPLEX_CMD(msg=True)
# solver = pulp.CPLEX_PY(msg=False)
# solver = pulp.PULP_CBC_CMD((msg=False)

# Solve the model 
m.solve(solver)

# Print optimal solutions if found
if pulp.LpStatus[m.status] == 'Optimal':
    print(f"Number of locations selected = {pulp.value(m.objective):,.2f}")
    for i in Nodes:
        if x[i].varValue  == 1:
            print(f"  Select location {i}")
