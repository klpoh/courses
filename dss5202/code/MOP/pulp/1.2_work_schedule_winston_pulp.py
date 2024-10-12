# -*- coding: utf-8 -*-
import pulp
""" Work force scheduing problem (Winston) PuLP (2024 10 12) """

# Sets or Indices
# The days of the week
Days =[ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Data
# req_data[d] = number of worker required on day d
req_data = [ 17, 13, 15, 19, 14, 16, 11 ]

# cover_matrix[i,j] = 1 if start-on-day j is available on day i, =0 otherwise
cover_matrix = [[1, 0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1]]

# Parameters Dictionaries
req = { day : r for day, r in zip(Days, req_data) }
cover = { (d1, d2) : cover_matrix[i][j] for i, d1 in enumerate(Days)
           for j, d2 in enumerate(Days) }

## The MIP Model
m = pulp.LpProblem("workforce_planning_problem", pulp.LpMinimize)

## Decision Variables
# x[d] = number of workers starting work on day d
x = { d : pulp.LpVariable(f'x({d})', lowBound=0, upBound=None, 
    cat=pulp.LpInteger) for d in Days }

## Objective Function
# Minimize the total number of workers employed
m += pulp.lpSum(x[d] for d in Days), 'Num_employed'

## Constraints
# The total number of available for each day d must be covered

for d in Days:
    m += pulp.lpSum(cover[d,j]*x[j] for j in Days) >= req[d], f'Cover({d})'

# Save model in LP format for inspection
# m.writeLP('workforce_planning.lp')

# Listof available solvers: pulp.listSolvers(True)
solver = pulp.GUROBI(msg=False)
# solver = pulp.GUROBI_CMD((msg=False)
# solver = pulp.GLPK_CMD(msg=True)
# solver = pulp.CPLEX_CMD(msg=True)
# solver = pulp.CPLEX_PY(msg=False)
# solver = pulp.PULP_CBC_CMD((msg=False)

# Solve the model 
m.solve(solver)

if pulp.LpStatus[m.status] == 'Optimal':
    print(f"Objective values = {pulp.value(m.objective)}")
    print("Number of people starting on day:")
    for d in Days:
        print(f"  {d}: {x[d].varValue}")
