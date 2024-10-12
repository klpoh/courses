# -*- coding: utf-8 -*-
import pulp
""" Product Mix Problem PuLP in Algebraic Format (2024 10 12) """

## Model
m = pulp.LpProblem("Product_Mix", pulp.LpMaximize)

## Sets or Indices
Products = ['door','window']
Plants = ['plant_1','plant_2','plant_3']

# Data (can be read from Excel file)
Plant_cap = [4, 12, 18]
TechCoeff = [[1, 0], [0, 2], [3, 2]]
Profit = [3, 5]

## Parameters
# Resource capacities
b = {plant : cap for plant, cap in zip(Plants, Plant_cap) }
A = {(plant, prod) : TechCoeff[i][j] 
     for i, plant in enumerate(Plants)
         for j, prod in enumerate(Products) }
c = {prod: profit for prod, profit in zip(Products, Profit)}

## Model
m = pulp.LpProblem("Product_mix", pulp.LpMaximize)

## Decision Variables
x = { p : pulp.LpVariable(f'x({p})', lowBound=0, upBound=None, 
    cat=pulp.LpContinuous) for p in Products }

# Objective function
m += pulp.lpSum(c[p]*x[p] for p in Products),'Profit'

## Constraints
for r in Plants:
    m += pulp.lpSum(A[r,p]*x[p] for p in Products) <= b[r], f'Capacity({r})'

# Save model for inspection/debugging
# m.writeLP('product_mix.lp')

# List of available solvers: pulp.listSolvers(True)
# solver = pulp.GUROBI(msg=False)
solver = pulp.GUROBI_CMD(msg=False)
# solver = pulp.GLPK_CMD(msg=True)
# solver = pulp.CPLEX_CMD(msg=True)
# solver = pulp.CPLEX_PY(msg=False)
# solver = pulp.PULP_CBC_CMD(msg=True)

# Solve the model
# m.solve()
m.solve(solver)

if pulp.LpStatus[m.status] == 'Optimal':
    print(f"Status: {pulp.LpStatus[m.status]}")
    print(f"Objective values = {m.objective.value():,.2f}")
    for v in m.variables():
        print(f"{v.name} = {v.varValue:.2f}")

