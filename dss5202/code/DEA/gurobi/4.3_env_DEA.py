# -*- coding: utf-8 -*-
# 4.3_env_DEA.py
import gurobipy as gp
from gurobipy import GRB
from math import isclose
"""  Environmental DEA (2024 10 20) """


def main():

    DMU = ['OECD', 'Middle East', 'Former USSR', 'Non-OECD Europe', 'China', 
           'Asia', 'Latin America', 'Africa']
    
    
    Inputs = ['Energy Consumption']
    D_Outputs = ['GDP']
    U_Outputs = ['CO2']
    
    X_data = [3696.5, 290.9, 610.17, 63.86,
              823.02, 851.4, 354.75, 404.42 ]
    
    Y_data = [25374.85, 1025.83, 1552.1, 358.26,
              5359.02, 5507.94, 2566.74, 1668.75 ]
    
    W_data = [12554.03, 1092.84, 2232.17, 252.84,
              3307.42, 2257.41, 844.61, 743.12 ]
    
    
    RTS = 'CRS'    # 'CRS', 'NIRS', 'VRS'
    env_DEA(DMU, Inputs, D_Outputs, U_Outputs, X_data, Y_data, W_data, RTS="CRS")



def env_DEA(DMU, Inputs, D_Outputs, U_Outputs, X_data, Y_data, W_data, RTS='CRS'):
    
    def env_DEA_CRS():
    # def env_DEA_CRS(DMU, Inputs, D_Outputs, U_Outputs, X_data, Y_data, W_data):
    
        print("\nEnvironmental DEA under CRS")
        # Parameters
        X = { (i, d) : v for i in Inputs for d, v in zip(DMU, X_data) }
        Y = { (r, d) : v for r in D_Outputs 
                         for d, v in zip(DMU, Y_data) }
        W = { (k, d) : v for k in U_Outputs 
                         for d, v in zip(DMU, W_data) }
        
        ## Store the optimization results
        ObjVal ={}
        isEff = {}
        Sminus = {}
        Splus = {}
        Lambda = {}
        RefSet = {}
        
        ## Solve a 2-phase optimization model for each DMU
        for d in DMU:
            m = gp.Model()
            # Decision variables
            tetha = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='tetha')
            lamda = m.addVars(DMU, vtype=GRB.CONTINUOUS, lb=0, name='lambda')
            sMinus = m.addVars(Inputs, vtype=GRB.CONTINUOUS, lb=0, name='s-')
            sPlus = m.addVars(D_Outputs, vtype=GRB.CONTINUOUS, lb=0, name='s+')
            
            # Phase I objective function for DMU r
            m.setObjective(tetha, GRB.MINIMIZE)
            
            # Constraints
            c1 = m.addConstrs(
                (gp.quicksum(X[i,j]*lamda[j] for j in DMU) + sMinus[i] == X[i,d]
                     for i in Inputs), name='c1')
            c2 = m.addConstrs(
                (gp.quicksum(Y[r,j]*lamda[j] for j in DMU) - sPlus[r] == Y[r,d] 
                     for r in D_Outputs), name='c2')
            c3 = m.addConstrs(
                (gp.quicksum(W[k,j]*lamda[j] for j in DMU) == tetha*W[k,d] 
                     for k in U_Outputs), name='c3')
            
           
            # Solve the Phase I model
            m.Params.OutputFlag = 0  # suppress all console messages
            m.optimize()
            if m.status != GRB.Status.OPTIMAL:
               print(f'DMU {d} is not optimal in Phase I')
            
            tetha_star = m.objVal
            
            # Phase II objective function
            m.setObjective(
                gp.quicksum(sMinus[i] for i in Inputs ) +
                gp.quicksum( sPlus[r] for r in D_Outputs), GRB.MAXIMIZE)
            
            # Replace the constraint c3
            m.remove(c3)
            c3r = m.addConstrs(
                (gp.quicksum(W[k,j]*lamda[j] for j in DMU) == tetha_star*W[k,d] 
                     for k in U_Outputs), name='c3r')
            
            
            # Solve the Phase II model
            m.optimize()
            if m.status != GRB.Status.OPTIMAL:
               print(f'DMU {d} is not optimal in Phase II')
        
            ObjVal[d]= tetha_star
            Sminus[d]= [sMinus[i].x for i in Inputs]
            Splus[d] = [sPlus[j].x for j in D_Outputs ]
            isEff[d] = isclose(tetha_star, 1) and \
                            all(s==0 for s in Sminus[d] + Splus[d])
            Lambda[d]= [lamda[k].x for k in DMU]
            RefSet[d]= [j for j in DMU if lamda[j].x > 0]
        
        ## Show the results
        for d in DMU:
            print(f'DMU {d}:')
            print(f'  Obj value = {ObjVal[d]:.4f}')
            print(f'  efficient = {isEff[d]}')
            print(f'      s- = {Sminus[d]}')
            print(f'      s+ = {Splus[d]}')
            print(f'  lambda = {Lambda[d]}')
            print(f'  RefSet = {RefSet[d]}')
    
    
    def env_DEA_NIRS():
    
        print("\nEnvironmental DEA under NIRS ")
        # Parameters
        X = { (i, d) : v for i in Inputs for d, v in zip(DMU, X_data) }
        Y = { (r, d) : v for r in D_Outputs 
                         for d, v in zip(DMU, Y_data) }
        W = { (k, d) : v for k in U_Outputs 
                         for d, v in zip(DMU, W_data) }
            
        
        ## Store the optimization results
        ObjVal ={}
        isEff = {}
        Sminus = {}
        Splus = {}
        Lambda = {}
        RefSet = {}
        
        ## Solve a 2-phase optimization model for each DMU
        for d in DMU:
            m = gp.Model()
            # Decision variables
            tetha = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='tetha')
            lamda = m.addVars(DMU, vtype=GRB.CONTINUOUS, lb=0, name='lambda')
            sMinus = m.addVars(Inputs, vtype=GRB.CONTINUOUS, lb=0, name='s-')
            sPlus = m.addVars(D_Outputs, vtype=GRB.CONTINUOUS, lb=0, name='s+')
            
            # Phase I objective function for DMU r
            m.setObjective(tetha, GRB.MINIMIZE)
            
            # Constraints
            c1 = m.addConstrs(
                (gp.quicksum(X[i,j]*lamda[j] for j in DMU) + sMinus[i] == X[i,d]
                     for i in Inputs), name='c1')
            c2 = m.addConstrs(
                (gp.quicksum(Y[r,j]*lamda[j] for j in DMU) - sPlus[r] == Y[r,d] 
                     for r in D_Outputs), name='c2')
            c3 = m.addConstrs(
                (gp.quicksum(W[k,j]*lamda[j] for j in DMU) == tetha*W[k,d] 
                     for k in U_Outputs), name='c3')
            c4 = m.addConstr(
                gp.quicksum(lamda[j] for j in DMU) <= 1, name='c4')
             
            
            # Solve the Phase I model
            m.Params.OutputFlag = 0  # suppress all console messages
            m.optimize()
            if m.status != GRB.Status.OPTIMAL:
               print(f'DMU {d} is not optimal in Phase I')
            
            tetha_star = m.objVal
            
            # Phase II objective function
            m.setObjective(
                gp.quicksum(sMinus[i] for i in Inputs ) +
                gp.quicksum( sPlus[r] for r in D_Outputs), GRB.MAXIMIZE)
            
            # Replace the constraint c3
            m.remove(c3)
            c3r = m.addConstrs(
                (gp.quicksum(W[k,j]*lamda[j] for j in DMU) == tetha_star*W[k,d] 
                     for k in U_Outputs), name='c3r')
            
            
            # Solve the Phase II model
            m.optimize()
            if m.status != GRB.Status.OPTIMAL:
               print(f'DMU {d} is not optimal in Phase II')
        
            ObjVal[d]= tetha_star
            Sminus[d]= [sMinus[i].x for i in Inputs]
            Splus[d] = [sPlus[j].x for j in D_Outputs ]
            isEff[d] = isclose(tetha_star, 1) and \
                            all(s==0 for s in Sminus[d] + Splus[d])
            Lambda[d]= [lamda[k].x for k in DMU]
            RefSet[d]= [j for j in DMU if lamda[j].x > 0]
        
        ## Show the results
        for d in DMU:
            print(f'DMU {d}:')
            print(f'  Obj value = {ObjVal[d]:.4f}')
            print(f'  efficient = {isEff[d]}')
            print(f'      s- = {Sminus[d]}')
            print(f'      s+ = {Splus[d]}')
            print(f'  lambda = {Lambda[d]}')
            print(f'  RefSet = {RefSet[d]}')
        
    
    def env_DEA_VRS():
        
        print("\nEnvironmental DEA under VRS ")
    
        # Parameters
        X = { (i, d) : v for i in Inputs for d, v in zip(DMU, X_data) }
        Y = { (r, d) : v for r in D_Outputs 
                         for d, v in zip(DMU, Y_data) }
        W = { (k, d) : v for k in U_Outputs 
                         for d, v in zip(DMU, W_data) }
            
        ## Store the optimization results
        ObjVal ={}
        isEff = {}
        Sminus = {}
        Splus = {}
        Lambda = {}
        RefSet = {}
        
        ## Solve a 2-phase optimization model for each DMU
        for d in DMU:
            m = gp.Model()
            # Decision variables
            tetha = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='tetha')
            lamda = m.addVars(DMU, vtype=GRB.CONTINUOUS, lb=0, name='lambda')
            sMinus = m.addVars(Inputs, vtype=GRB.CONTINUOUS, lb=0, name='s-')
            sPlus = m.addVars(D_Outputs, vtype=GRB.CONTINUOUS, lb=0, name='s+')
            beta = m.addVar(vtype=GRB.CONTINUOUS, ub=1, name='beta')
            
            # Phase I objective function for DMU r
            m.setObjective(tetha, GRB.MINIMIZE)
            
            # Constraints
            c1 = m.addConstrs(
                (gp.quicksum(X[i,j]*lamda[j] for j in DMU) + sMinus[i] == beta*X[i,d]
                     for i in Inputs), name='c1')
            c2 = m.addConstrs(
                (gp.quicksum(Y[r,j]*lamda[j] for j in DMU) - sPlus[r] == Y[r,d] 
                     for r in D_Outputs), name='c2')
            c3 = m.addConstrs(
                (gp.quicksum(W[k,j]*lamda[j] for j in DMU) == tetha*W[k,d] 
                     for k in U_Outputs), name='c3')
            c4 = m.addConstr(
                gp.quicksum(lamda[j] for j in DMU) == beta, name='c4')
             
            
            # Solve the Phase I model
            m.Params.OutputFlag = 0  # suppress all console messages
            m.optimize()
            if m.status != GRB.Status.OPTIMAL:
               print(f'DMU {d} is not optimal in Phase I')
            
            tetha_star = m.objVal
            beta_star = beta.x
            
            # Phase II objective function
            m.setObjective(
                gp.quicksum(sMinus[i] for i in Inputs ) +
                gp.quicksum( sPlus[r] for r in D_Outputs), GRB.MAXIMIZE)
            
            # Replace constraint c1
            m.remove(c1)
            c1r = m.addConstrs(
                (gp.quicksum(X[i,j]*lamda[j] for j in DMU) + sMinus[i] == beta_star*X[i,d]
                     for i in Inputs), name='c1r')
            
            
            # Replace constraint c3
            m.remove(c3)
            c3r = m.addConstrs(
                (gp.quicksum(W[k,j]*lamda[j] for j in DMU) == tetha_star*W[k,d] 
                     for k in U_Outputs), name='c3r')
            
            # Replace constraint c4
            m.remove(c4)
            c4r = m.addConstr(
                gp.quicksum(lamda[j] for j in DMU) == beta_star, name='c4r')
             
            
            # Solve the Phase II model
            m.optimize()
            if m.status != GRB.Status.OPTIMAL:
               print(f'DMU {d} is not optimal in Phase II')
        
            ObjVal[d]= tetha_star
            Sminus[d]= [sMinus[i].x for i in Inputs]
            Splus[d] = [sPlus[j].x for j in D_Outputs ]
            isEff[d] = isclose(tetha_star, 1) and \
                            all(s==0 for s in Sminus[d] + Splus[d])
            Lambda[d]= [lamda[k].x for k in DMU]
            RefSet[d]= [j for j in DMU if lamda[j].x > 0]
        
        
        ## Show the results
        for d in DMU:
            print(f'DMU {d}:')
            print(f'  Obj value = {ObjVal[d]:.4f}')
            print(f'  efficient = {isEff[d]}')
            print(f'      s- = {Sminus[d]}')
            print(f'      s+ = {Splus[d]}')
            print(f'  lambda = {Lambda[d]}')
            print(f'  RefSet = {RefSet[d]}')

    if RTS == 'CRS':
        env_DEA_CRS()
    elif RTS == 'NIRS':
        env_DEA_NIRS()
    elif RTS == 'VRS':
        env_DEA_VRS()
    else:
        print("Invalid RTS parameter")


if __name__=="__main__":
    main()

