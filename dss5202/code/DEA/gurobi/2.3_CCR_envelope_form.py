# -*- coding: utf-8 -*-
# 2.3_CCR_envelope_form.py
import gurobipy as gp
from gurobipy import GRB
from math import isclose
"""  CCR Model in Envelope Form 2024 10 20 """

def main():
    
    # Uncomment the example you want to run
    one_input_one_output()
    # two_input_one_output()
    # one_input_two_output()
    # two_input_two_output()
    pass


def one_input_one_output():
    """ one input one output example """
    
    DMU = ['A','B','C','D','E','F','G','H']
    Inputs = ['Employee']
    Outputs = ['Sale']
    
    Inputs_data = {'A' : [ 2 ],
                   'B' : [ 3 ],
                   'C' : [ 3 ],
                   'D' : [ 4 ],
                   'E' : [ 5 ],
                   'F' : [ 5 ],
                   'G' : [ 6 ],
                   'H' : [ 8 ] }
    
    Outputs_data = {'A' : [ 1 ],
                    'B' : [ 3 ],
                    'C' : [ 2 ],
                    'D' : [ 3 ],
                    'E' : [ 4 ],
                    'F' : [ 2 ],
                    'G' : [ 3 ],
                    'H' : [ 5 ] }
                
    CCR_envelope(DMU, Inputs, Outputs, Inputs_data, Outputs_data)


def two_input_one_output():
    """ two inputs one output example """
    
    DMU = ['A','B','C','D','E','F','G','H', 'I']
    Inputs = ['Employee','Area']
    Outputs = ['Sale']
    
    Inputs_data = {'A' : [ 4, 3 ],
                   'B' : [ 7, 3 ],
                   'C' : [ 8, 1 ],
                   'D' : [ 4, 2 ],
                   'E' : [ 2, 4 ],
                   'F' : [ 5, 2 ],
                   'G' : [ 6, 4 ],
                   'H' : [5.5, 2.5],
                   'I' : [ 6, 2.5 ]}
    
    Outputs_data = {'A' : [ 1 ],
                    'B' : [ 1 ],
                    'C' : [ 1 ],
                    'D' : [ 1 ],
                    'E' : [ 1 ],
                    'F' : [ 1 ],
                    'G' : [ 1 ],
                    'H' : [ 1 ],
                    'I' : [ 1 ] }
                
    CCR_envelope(DMU, Inputs, Outputs, Inputs_data, Outputs_data)


def one_input_two_output():
    """ one input two outputs example """
    
    DMU = ['A','B','C','D','E','F','G']
    Inputs = ['Employee']
    Outputs = ['Customer', 'Sale']
    
    Inputs_data = {'A' : [ 1 ],
                   'B' : [ 1 ],
                   'C' : [ 1 ],
                   'D' : [ 1 ],
                   'E' : [ 1 ],
                   'F' : [ 1 ],
                   'G' : [ 1 ]  }
    
    Outputs_data = {'A': [ 1, 5 ],
                    'B': [ 2, 7 ],
                    'C': [ 3, 4 ],
                    'D': [ 4, 3 ],
                    'E': [ 4, 6 ],
                    'F': [ 5, 5 ],
                    'G': [ 6, 2 ] } 

    CCR_envelope(DMU, Inputs, Outputs, Inputs_data, Outputs_data)


def two_input_two_output():
    """ two inputs two outputs example """
    
    DMU = ['A','B','C','D','E','F','G','H','I','J','K','L']
    Inputs = ['Doctor','Nurse']
    Outputs = ['Outpatient', 'Inpatient']
    
    Inputs_data = {'A' : [ 20, 151 ],
                   'B' : [ 19, 131 ],
                   'C' : [ 25, 160 ],
                   'D' : [ 27, 168 ],
                   'E' : [ 22, 158 ],
                   'F' : [ 55, 255 ],
                   'G' : [ 33, 235 ],
                   'H' : [ 31, 206 ],
                   'I' : [ 30, 244 ],
                   'J' : [ 50, 268 ],
                   'K' : [ 53, 306 ],
                   'L' : [ 38, 284 ] }
    
    Outputs_data = {'A': [ 100,  90 ],
                    'B': [ 150,  50 ],
                    'C': [ 160,  55 ],
                    'D': [ 180,  72 ],
                    'E': [  94,  66 ],
                    'F': [ 230,  90 ],
                    'G': [ 220,  88 ],
                    'H': [ 152,  80 ],
                    'I': [ 190, 100 ],
                    'J': [ 250, 100 ],
                    'K': [ 260, 147 ],
                    'L': [ 250, 120 ] } 

    CCR_envelope(DMU, Inputs, Outputs, Inputs_data, Outputs_data)



def CCR_envelope(DMU, Inputs, Outputs, Inputs_data, Outputs_data):

    # Parameters
    X = {(invar,dmu) : v  for dmu, values in Inputs_data.items()
         for invar, v in zip(Inputs, values) }

    Y = {(outvar,dmu) : v  for dmu, values in Outputs_data.items()
         for outvar, v in zip(Outputs, values) }
    
    ## Store the optimization results
    ObjVal ={}
    isEff = {}
    Sminus = {}
    Splus = {}
    Lambda = {}
    RefSet = {}
    
    ## Solve a 2-phase optimization model for each DMU
    for r in DMU:
        m = gp.Model()
        # Decision variables
        tetha = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='tetha')
        lamda = m.addVars(DMU, vtype=GRB.CONTINUOUS, lb=0, name='lambda')
        sMinus = m.addVars(Inputs, vtype=GRB.CONTINUOUS, lb=0, name='s-')
        sPlus = m.addVars(Outputs, vtype=GRB.CONTINUOUS, lb=0, name='s+')
        
        # Phase I objective function for DMU r
        m.setObjective(tetha, GRB.MINIMIZE)
        
        # Constraints
        c1 = m.addConstrs(
            (gp.quicksum(X[i,k]*lamda[k] for k in DMU) + sMinus[i] == X[i,r]*tetha
                 for i in Inputs), name='c1')
        c2 = m.addConstrs(
            (gp.quicksum(Y[j,k]*lamda[k] for k in DMU) - sPlus[j] == Y[j,r] 
                 for j in Outputs), name='c2')
        
        
        # Solve the Phase I model
        m.Params.OutputFlag = 0  # suppress all console messages
        m.optimize()
        if m.status != GRB.Status.OPTIMAL:
           print(f'DMU {r} is not optimal in Phase I')
        
        tetha_star = m.objVal
        
        # Phase II objective function
        m.setObjective(
            gp.quicksum(sMinus[i] for i in Inputs ) +
            gp.quicksum( sPlus[j] for j in Outputs), GRB.MAXIMIZE)
        
        # Replace the constraint c1 with c3
        m.remove(c1)
        c3 = m.addConstrs(
            (gp.quicksum(X[i,k]*lamda[k] for k in DMU) + sMinus[i] == 
                 X[i,r]*tetha_star for i in Inputs), name='c3')
        # Solve the Phase II model
        m.optimize()
        if m.status != GRB.Status.OPTIMAL:
           print(f'DMU {r} is not optimal in Phase II')
    
        ObjVal[r]= tetha_star
        Sminus[r]= [sMinus[i].x for i in Inputs]
        Splus[r] = [sPlus[j].x for j in Outputs ]
        isEff[r] = isclose(tetha_star, 1) and \
                        all(s==0 for s in Sminus[r] + Splus[r])
        Lambda[r]= [lamda[k].x for k in DMU]
        RefSet[r]= [k for k in DMU if lamda[k].x > 0]
    
    ## Show the results
    for d in DMU:
        print(f'DMU {d}:')
        print(f'  Obj value = {ObjVal[d]}')
        print(f'  efficient = {isEff[d]}')
        print(f'      s- = {Sminus[d]}')
        print(f'      s+ = {Splus[d]}')
        print(f'  lambda = {Lambda[d]}')
        print(f'  RefSet = {RefSet[d]}')

if __name__=="__main__":
    main()
