#Import libraries
import pandas as pd
import itertools 
import numpy as np
import random
import networkx as nx
import json


import matplotlib as mpl
import matplotlib.pyplot as plt

from docplex.mp.model import Model
from docplex.mp.vartype import VarType
# from docplex.mp.basic import ObjectiveSense

#P set of all requests: pi = (si, di, ai ) P is the set of all the requests {0,...,P}.
#f_i_t_u_v 
#T set of all timeslots

W_adj = np.array([[0, 30, 40, 10], [60, 0, 30, 0], [0, 24, 0, 30], [10, 0, 0, 0]])  #shoould give 0-1-2-3-0, 0-3-0 and 0-3-0
P_set = [(0, 1, 0), (2, 3, 5)]
P = len(P_set)
n , e = len(W_adj), len(np.nonzero(W_adj)[0])
print(n, e)
T = 6

m = Model(name='vrp',log_output=False)
m.objective_sense = 'max'
m.parameters.threads.set(1)

####VARIABLES####
variables = m.binary_var_dict([(i, t, u, v) for i in range(P) for t in range(T) for u in range(n) for v in range(n) if W_adj[u][v] != 0], name='x')
print(variables)

####OBJECTIVE####
objective = 0
for i in range(P):
    for t in range(T):
        for w in range(n):
            if W_adj[P_set[i][0]][w] != 0:
                print((i, t, P_set[i][0], w), variables[(i, t, P_set[i][0], w)])
                objective += variables[(i, t, P_set[i][0], w)]
                # print(variables[(i, t, P_set[i][0], w)])
print(objective)
m.maximize(objective)

####CONSTRAINTS####

#At every timestep, sum of all demands on an edge should be <= capacity of edge i.e 1 in this paper
for t in range(T):
    for u in range(n):
        for v in range(n):
            if W_adj[u][v] != 0:       #This is an edge
                sum_allocated = 0
                for i in range(P):
                    sum_allocated += variables[(i, t, u, v)]
                m.add_constraint(sum_allocated <= 1, 'edge_capacity_'+str(u)+str(v)+'_t_'+str(t))
                print(m.get_constraint_by_name('edge_capacity_'+str(u)+str(v)+'_t_'+str(t)))

#No allocation should begin for a request before arrival time
for i in range(P):
    sum_allocated = 0
    for t in range(0, P_set[i][2]):
        for u in range(n):
            for v in range(n):
                if W_adj[u][v] != 0:
                    sum_allocated += variables[(i, t, u, v)]
    m.add_constraint(sum_allocated == 0, 'no_allocation_bef_arrival_req_'+str(i))
    print(m.get_constraint_by_name('no_allocation_bef_arrival_req_'+str(i)))

#A request is served only once:
for i in range(P):
    for u in range(n):
        for v in range(n):
            if W_adj[u][v] != 0:
                sum_allocated = 0
                for t in range(P_set[i][2], T):
                    sum_allocated += variables[(i, t, u, v)]
                m.add_constraint(sum_allocated <= 1, 'req_served_once_'+str(i)+'_edge_'+str(u)+str(v))
                print(m.get_constraint_by_name('req_served_once_'+str(i)+'_edge_'+str(u)+str(v)))    

#Flow conservation at other than src and dst
for i in range(P):
    for t in range(T):
        for v in range(n):
            inflow_v = 0
            outflow_v = 0
            for u in range(n):
                if W_adj[u][v] != 0:
                    inflow_v += variables[(i, t, u, v)]
            # print(inflow_v)
            for w in range(n):
                if W_adj[v][w] != 0:
                    outflow_v += variables[(i, t, v, w)]
            # print(outflow_v)
            if v not in (P_set[i][0], P_set[i][1]): #Flow should be conserved at middle nodes
                m.add_constraint((inflow_v - outflow_v) == 0, 'flow_cons_node_'+str(v)+'_t_'+str(t)+'_req_'+str(i))
                print(m.get_constraint_by_name('flow_cons_node_'+str(v)+'_t_'+str(t)+'_req_'+str(i)))

#Flow cons at src
for i in range(P):
    for t in range(T):
        inflow_src = 0
        outflow_src = 0
        src = P_set[i][0]
        for u in range(n): 
            if W_adj[u][src] != 0:
                inflow_src += variables[(i, t, u, src)]
        
        for w in range(n):
            if W_adj[src][w] != 0:
                outflow_src += variables[(i, t, src, w)]
        
        # m.add_constraint(outflow_src - inflow_src <=1, 'flow_cons_src'+str(src))
        m.add_constraint(outflow_src <=1, 'flow_cons_src'+str(src)+'_t_'+str(t))
        print(m.get_constraint_by_name('flow_cons_src'+str(src)+'_t_'+str(t)))

#Flow cons at dst
for i in range(P):
    for t in range(T):
        inflow_dst = 0
        outflow_dst = 0
        dst = P_set[i][1]
        for u in range(n): 
            if W_adj[u][dst] != 0:
                inflow_dst += variables[(i, t, u, dst)]
        
        for w in range(n):
            if W_adj[dst][w] != 0:
                outflow_dst += variables[(i, t, dst, w)]
        
        # m.add_constraint(inflow_dst - outflow_dst <=1, 'flow_cons_dst'+str(dst))
        m.add_constraint(inflow_dst<=1, 'flow_cons_dst'+str(dst)+'_t_'+str(t))
        print(m.get_constraint_by_name('flow_cons_dst'+str(dst)+'_t_'+str(t)))

solution = m.solve()
sol_json = solution.export_as_json_string()
print('----------Problem Details: -----------')
print(sol_json)
print('----------Solution Details: -----------')
vars = json.loads(sol_json)['CPLEXSolution']['variables']
for v in vars:
    print(v['name'])





