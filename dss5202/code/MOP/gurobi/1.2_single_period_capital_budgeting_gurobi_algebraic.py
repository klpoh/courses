# -*- coding: utf-8 -*-
# 1.2_single_period_capital_budgetting_gurobi_algebraic.py
import gurobipy as gp
from gurobipy import GRB
""" Single Period Budgetting Problem Gurobi Algebric (2024 10 06) """

## Sets
Investments =['Inv1','Inv2','Inv3','Inv4','Inv5']
Periods = [0, 1]

## Data
Expenses = [[11, 53,  5,  5, 29 ],
            [ 3,  6,  5,  1, 34 ]]
NPV = [13, 16, 16, 14, 39 ]
Budgets = [40, 20 ]

## Parameters
exp = { (per, inv) : Expenses[j][i] 
           for i, inv in enumerate(Investments)
               for j, per in enumerate(Periods) }
npv  = { i : e for i, e in zip(Investments, NPV) }
bgt = { p : b for p, b in zip(Periods, Budgets) }

## Model
m = gp.Model('single_period_capital_budgetting')

## Decision Variables
x = m.addVars(Investments, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
# x = m.addVars(Investments, vtype=GRB.BINARY, name="x")

# Objective function
m.setObjective(
     gp.quicksum(npv[i]*x[i] for i in Investments), GRB.MAXIMIZE)
    
## Constraints
m.addConstrs(
    (gp.quicksum(exp[p,i]*x[i] for i in Investments) <= bgt[p]
         for p in Periods), name='Budget')

# Save model for inspection/debugging
# m.write('single_period_budget.lp')

# Solve the model
m.optimize()

# Print optimal solutions
if m.status == GRB.Status.OPTIMAL:
    print("\nOptimal Solution:")
    print(f"Profit = {m.objVal:.6f}")
    for i in Investments:
        print(f"{i} = {x[i].x:.6f}")
