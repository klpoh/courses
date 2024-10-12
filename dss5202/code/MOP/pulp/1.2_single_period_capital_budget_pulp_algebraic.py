# -*- coding: utf-8 -*-
import pulp
""" Single Period Capital Budgetting Pulp (2024 10 12) """

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
m = pulp.LpProblem("single_period_capital", pulp.LpMaximize)

## Decision Variables
x = { i : pulp.LpVariable(f'x({i})', lowBound=0, upBound=1, 
    cat=pulp.LpContinuous) for i in Investments }

## Objective function
m += pulp.lpSum(npv[i]*x[i] for i in Investments),'NPV'

## Constraints
for p in Periods:
    m += pulp.lpSum(exp[p,i]*x[i] for i in Investments) <= bgt[p], f'budget({p})'

# Save model for inspection/debugging
# m.writeLP('single_period_budget.lp')

# List of available solvers: pulp.listSolvers(True)
# solver = pulp.GUROBI(msg=False)
# solver = pulp.GUROBI_CMD(msg=False)
# solver = pulp.GLPK_CMD(msg=False)
# solver = pulp.CPLEX_CMD(msg=True)
# solver = pulp.CPLEX_PY(msg=False)
# solver = pulp.PULP_CBC_CMD(msg=False)

# Solve the model
m.solve()   # Use default solver
# m.solve(solver)

if pulp.LpStatus[m.status] == 'Optimal':
    print(f"Status: {pulp.LpStatus[m.status]}")
    print(f"Objective values = {m.objective.value():,.6f}")
    for v in m.variables():
        print(f"{v.name} = {v.varValue:.6f}")


