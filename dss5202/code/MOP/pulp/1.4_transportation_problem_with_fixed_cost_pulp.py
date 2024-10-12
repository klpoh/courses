# -*- coding: utf-8 -*-
import pulp
""" Transportation problem with fixed costs PuLP (2024 10 12) """

# Sets or Indices
Plants = ['P1','P2','P3','P4','P5']
Customers = ['W1','W2','W3','W4']

## Data. They can be read from external file
cap_data = [20, 22, 17, 19, 18]
fCosts_data = [12000, 15000, 17000, 13000, 16000]
demands_data    = [15, 18, 14, 20]
tranCosts_data  = [[4000, 2500, 1200, 2200],
                   [2000, 2600, 1800, 2600],
                   [3000, 3400, 2600, 3100],
                   [2500, 3000, 4100, 3700],
                   [4500, 4000, 3000, 3200]]

## Parameters
# Dictionaries to enable data to be indexed by Plants and Customers
cap = {p: c for p, c in zip(Plants, cap_data)}

demand = {c: d for c, d in zip(Customers, demands_data)}

transCost = {(p, c): tranCosts_data[i][j] 
             for i, p in enumerate(Plants)
                 for j, c in enumerate(Customers)}

fCost = {p : fc for p, fc in zip(Plants, fCosts_data)}
         
# Model
m = pulp.LpProblem("Trans_Problem_fixed_cost", pulp.LpMinimize)

# Decision Variables
x = {(i,j): pulp.LpVariable(f'x({i},{j})', lowBound=0, upBound=None, 
    cat=pulp.LpContinuous) for i in Plants for j in Customers}

y = {i: pulp.LpVariable(f'y({i})', cat=pulp.LpBinary) for i in Plants}

# Objective function 
m += pulp.lpSum(x[i,j]*transCost[i,j] for i in Plants for j in Customers) + \
     pulp.lpSum(y[i]*fCost[i] for i in Plants), 'Total_Costs'

# Constraints
# Supplies from each plant p are limited by its capacity and if open
for i in Plants:
    m += pulp.lpSum(x[i,j] for j in Customers) <= cap[i]*y[i], f'Capacity({i})'
    
# Demand for warehouses
for j in Customers:
    m += pulp.lpSum(x[i,j] for i in Plants) == demand[j], f"Demand({j})"

         
# Write the model to an .lp file for inspection
# m.writeLP("Trans_fixed_cost.lp")

# Listof available solvers: pulp.listSolvers(True)
# solver = pulp.GUROBI(msg=False)
# solver = pulp.GUROBI_CMD(msg=False)
# solver = pulp.GLPK_CMD(msg=False)
# solver = pulp.CPLEX_CMD(msg=False)
# solver = pulp.CPLEX_PY(msg=False)
solver = pulp.PULP_CBC_CMD(msg=False)

# Solve the model 
m.solve(solver)
# m.solve()

print(f"Status: {pulp.LpStatus[m.status]}")

if pulp.LpStatus[m.status] == 'Optimal':
    print(f"Objective values = {pulp.value(m.objective):,.2f}")
    
    # Decison variables values and reduced costs
    print("\nDecision Variables:")
    for v in m.variables():
        print(f"  {v.name:}= {v.varValue}, reduced cost= {v.dj}")
    
    print("\nConstraints:")
    for name, c in list(m.constraints.items()):
        print(f"  {name:}: slack= {c.slack}, dual price = {c.pi}")
