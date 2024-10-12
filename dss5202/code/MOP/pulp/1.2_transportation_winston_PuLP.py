# -*- coding: utf-8 -*-
import pulp
""" Transportation problem Winston (2024 10 12) """

# Sets or Indices
Plants = ['P1','P2','P3']
Cities = ['C1','C2','C3','C4']

## Data. They can be read from external file
supply_data = [35, 50, 40]
demand_data = [45, 20, 30, 30]
cost_data  = [[8,  6, 10, 9],
               [9, 12, 13, 7],
               [14, 9, 16, 5]]

## Parameters.
num_plants = len(Plants)
num_customers = len(Cities)
# Dictionaries to enable data to be indexed by plants and cities
Supply = { p : s for p, s in zip(Plants, supply_data)}
Demand = { c : d for c, d in zip(Cities, demand_data)}
Cost = {(p, c): cost_data[i][j] 
                  for i, p in enumerate(Plants)
                  for j, c in enumerate(Cities)}

## Model
m = pulp.LpProblem("Transportation_Problem_Winston_Example", pulp.LpMinimize)

## Decision Variables
# x[(i,j)] = quantity shipped from plant i to city j
x = {(i,j): pulp.LpVariable(f'x({i},{j})', lowBound=0, upBound=None, 
    cat=pulp.LpContinuous) for i in Plants for j in Cities}

# Objective function
# Minimize the total cost
m += pulp.lpSum(x[i,j]*Cost[i,j] for i in Plants for j in Cities),'Total_Costs'

## Constraints
# Supplies from each plant i are limited by its supply capacity
for i in Plants:
    m += pulp.lpSum(x[i,j] for j in Cities) <= Supply[i], f'Supply({i})'

# Demands at each city j must be met
for j in Cities:
    m += pulp.lpSum(x[i,j] for i in Plants) == Demand[j], f"Demand({j})"

# Save model for inspection/debugging
m.writeLP('Transportation_problem_winston.lp')

# Solve the model

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

    for i in Plants:
        for j in Cities:
            print(f"Ship {x[i,j].varValue:4.1f} from {i} to {j}")
