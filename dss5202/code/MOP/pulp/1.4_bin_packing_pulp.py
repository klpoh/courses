# -*- coding: utf-8 -*-
import pulp
""" Bin Packing Problem Pulp (2024 10 12) """

# Sets
Items = range(1, 10)
Bins = range(1, 10)

# Data
Sizes = [0.5, 0.7, 0.5, 0.2, 0.4, 0.2, 0.5, 0.1, 0.6]

# Parameters
s = { item : size for item, size in zip(Items, Sizes) }

## Model
m = pulp.LpProblem("Bin_Packing_Probem", pulp.LpMinimize)

## Decision Variables
x = {(i,j): pulp.LpVariable(f'x({i},{j})', cat=pulp.LpBinary) 
     for i in Items for j in Bins}
y = { j : pulp.LpVariable(f'y{j})', cat=pulp.LpBinary) for j in Bins}

# Objective function
m += pulp.lpSum(y[j] for j in Bins),'No_of_bins'

## Constraints
for j in Bins:
    m += pulp.lpSum(s[i]*x[i,j] for i in Items) <= y[j], f'Bin_cap({j})'
    
for i in Items:
    m += pulp.lpSum(x[i,j] for j in Bins) == 1, f'Assign({i})'

# Save model for inspection/debugging
# m.writeLP('BPP.lp')

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
    print(f"Obj values = {pulp.value(m.objective)} bins")

    k = 0
    for j in Bins:
        if y[j].varValue == 1:
            k +=1
            print(f"Bin {k}:")
            for i in Items:
                if x[i,j].varValue == 1:
                    print(f" Item {i} ({s[i]})")
