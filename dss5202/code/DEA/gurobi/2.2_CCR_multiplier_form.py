# -*- coding: utf-8 -*-
# 2.2_CCR_multiplier_form.py
import gurobipy as gp
from gurobipy import GRB
"""  CCR Model in Multipler Form (2024 10 20) """

def main():
    
    # Uncomment the example you want to run
    # one_input_one_output()
    # two_input_one_output()
    # one_input_two_output()
    two_input_two_output()
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
                
    CCR_multiplier(DMU, Inputs, Outputs, Inputs_data, Outputs_data)


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
                
    CCR_multiplier(DMU, Inputs, Outputs, Inputs_data, Outputs_data)


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

    CCR_multiplier(DMU, Inputs, Outputs, Inputs_data, Outputs_data)


    
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

    CCR_multiplier(DMU, Inputs, Outputs, Inputs_data, Outputs_data)



def CCR_multiplier(DMU, Inputs, Outputs, Inputs_data, Outputs_data ):

    # Parameters
    X = {(invar,dmu) : v  for dmu, values in Inputs_data.items()
         for invar, v in zip(Inputs, values) }

    Y = {(outvar,dmu) : v  for dmu, values in Outputs_data.items()
         for outvar, v in zip(Outputs, values) }

    ObjVal ={}
    isEff = {}
    ustar = {}
    vstar = {}
    RefSet = {}
    
    for r in DMU:
        m = gp.Model()
        # Decision variables
        v = m.addVars(Inputs,  vtype=GRB.CONTINUOUS, lb=0, name='v')
        u = m.addVars(Outputs, vtype=GRB.CONTINUOUS, lb=0, name='u')
        
        # Objective function
        m.setObjective(
            gp.quicksum(u[j]*Y[j,r] for j in Outputs), GRB.MAXIMIZE)
        
        # Constraints
        m.addConstr(
            gp.quicksum(v[j]*X[j,r] for j in Inputs) == 1, name='c1')
        
        c2=m.addConstrs(
            (gp.quicksum(u[j]*Y[j,d] for j in Outputs) - \
                gp.quicksum(v[j]*X[j,d] for j in Inputs) <= 0
                    for d in DMU), name='c2')
        
        m.Params.OutputFlag = 0
        m.optimize()
        if m.status != GRB.Status.OPTIMAL:
           print(f'\nDMU {r} is not optimal')
        else:
            ObjVal[r] = m.objVal
            isEff[r] = m.objVal == 1
            vstar[r] = [ v[j].x for j in Inputs ]
            ustar[r] = [ u[j].x for j in Outputs ]
            RefSet[r] =  [ d for d in DMU if c2[d].getAttr("slack") == 0 ]
    
    
    Eff_DMU = [ dmu for dmu in DMU if isEff[dmu]]
    print(f"\nEfficient DMU = {Eff_DMU}")
    for d in DMU:
        print(f"DMU {d}")
        print(f"  Eff = {ObjVal[d]:.4f},  {isEff[d]}")
        print(f"  Ref Set = {RefSet[d]}")
        print(f"  v = {vstar[d]}")
        print(f"  u = {ustar[d]}")
    
          
if __name__=="__main__":
    main()
    