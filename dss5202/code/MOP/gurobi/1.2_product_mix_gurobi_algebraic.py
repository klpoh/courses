# -*- coding: utf-8 -*-
# product_mix_gurobi_algebraic.py
import gurobipy as gp
from gurobipy import GRB
""" Product Mix Problem in Algebraic Model Format (2024 09 08) """

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
m = gp.Model('product_mix')

## Decision Variables
# Quantity of product produced
x = m.addVars(Products, vtype=GRB.CONTINUOUS, lb=0, name="x")

# Objective function
# Max total profits
m.setObjective(
    gp.quicksum(c[prod]*x[prod] for prod in Products), GRB.MAXIMIZE)
    
## Constraints
# Capacity of each resource cannot be exceeded.
m.addConstrs(
    (gp.quicksum(A[plant,prod]*x[prod] for prod in Products) <= b[plant] 
         for plant in Plants), name='Capacity')

# Save model for inspection/debugging
# m.write('product_mix.lp')

# Solve the model
m.optimize()

# Print optimal solutions
if m.status == GRB.Status.OPTIMAL:
    print("\nOptimal Solution:")
    print(f"Profit = {m.objVal}")
    
    for i in Products:
        print(f"Product {i} = {x[i].x}")

