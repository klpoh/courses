# -*- coding: utf-8 -*-
import pulp
""" Multi-Period Capital Budgeting Problem (2024 10 12) """

# Sets or Indices
Projects = ['P1','P2','P3','P4','P5']
Years = [1, 2, 3]
Budgets = [25, 25, 25]
Cash_flows = [[5, 1, 8 ],
              [4, 7, 10],
              [3, 9, 2 ],
              [7, 4, 1 ],
              [8, 6, 10]]
Benefits = [20, 40, 20, 15, 30 ]

## Parameters
budget = {y: b for y, b in zip(Years, Budgets)}
cf = {(p, y) : Cash_flows[i][j] 
                for i, p in enumerate(Projects)
                    for j, y in enumerate(Years)}
npv = {p: b for p, b in zip(Projects, Benefits)}

## Model
m = pulp.LpProblem("multi_period_budget", pulp.LpMaximize)

## Decision Variables
x = { j : pulp.LpVariable(f'x({j})', cat=pulp.LpBinary) for j in Projects}

# Objective function
m += pulp.lpSum(npv[i]*x[i] for i in Projects),'NPV'

## Constraints
for j in Years:
    m += pulp.lpSum(cf[i,j]*x[i] for i in Projects) <= budget[j], f'Budget({j})'

# Save model for inspection/debugging
# m.writeLP('multi_period_budget.lp')

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
    print(f"Objective values = {pulp.value(m.objective):,.2f}")
    print("Select:")
    for i in Projects:
        if x[i].varValue == 1:
            print(f"  Project {i}:")
