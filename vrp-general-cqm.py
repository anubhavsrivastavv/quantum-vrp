#Import libraries
import pandas as pd
import itertools 
import numpy as np
import random
import networkx as nx


import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict 
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler, LazyFixedEmbeddingComposite
from dwave_qbsolv import QBSolv
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler
from greedy import SteepestDescentSolver
from dimod import ConstrainedQuadraticModel, Integer, Binary
from dwave.system import LeapHybridCQMSampler
import time

def plot_graph(G, color_map):
    pos=nx.spring_layout(G) 
    nx.draw_networkx(G, pos, node_color= 'white')
    labels = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.33)
    plt.show()

# W_adj = np.array([[0, 30, 40, 10], [60, 0, 30, 0], [0, 24, 0, 30], [10, 0, 0, 0]])  #shoould give 0-1-2-3-0, 0-3-0 and 0-3-0
# n, K, R = 4, 3, 3

W_adj = np.array([[0,5,0,0,5], [10,0,5,5,0], [0,0,0,5,0], [2,0,0,0,20], [10,0,0,5,0]])
n, K = 5, 3

W_adj = np.array([[0, 5, 0, 6, 0, 0, 0, 0], [1, 0, 3, 0, 0, 0, 0, 0], [6, 0, 0, 4, 0, 8, 0, 5], [0, 0, 0, 0, 8, 15, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 9, 0], [0, 0, 0, 0, 0, 0, 0, 3], [3, 0, 0, 0, 0, 0, 0, 0]])
n, K = 8, 3
e = len(np.nonzero(W_adj)[0])
print(e)


color_map = ['white' for i in range(n)]
A, B_penalty, C_penalty, D_penalty = 1000, 1000, 1000, 1000


#Visualize the graph
labels = [i for i in range(len(W_adj))]     #To name the rows and cols of the pandas dataframe
A2 = pd.DataFrame(W_adj, index=labels, columns=labels)

G_base = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())
plot_graph(G_base, color_map)


i = Integer("i")
cqm = ConstrainedQuadraticModel()

####VARIABLES####
variables = {}
for k in range(K):
    for i in range(n):
        for j in range(n):
            if W_adj[i][j] != 0:
                variables[(k, i, j)] = Binary(f'x_{k}_{i}_{j}')
                # cqm.add_variable('BINARY', f'x_{k}_{v}_{t}')

print(f'{cqm.variables}')

####OBJECTIVE####
objective = 0
for k in range(K):
    for i in range(n):
        for j in range(n):
            if W_adj[i][j] != 0:
                objective += variables[(k, i, j)]*W_adj[i][j]
print()
print(objective)

cqm.set_objective(objective)

####CONSTRAINTS####
def outgoing_variables(vertex, k = None):
    to = [j for j in range(len(W_adj[vertex])) if W_adj[vertex][j] != 0]
    if k is None:
        vars = [variables[(k, vertex, to_vertex)] for k in range(K) for to_vertex in to]
    else:
        vars = [variables[(k, vertex, to_vertex)] for to_vertex in to]
    return vars

def incoming_variables(vertex, k = None):
    from_ = [i for i in range(len(W_adj)) if W_adj[i][vertex] != 0]
    if k is None:
        vars = [variables[(k, from_vertex, vertex)] for k in range(K) for from_vertex in from_]
    else:
        vars = [variables[(k, from_vertex, vertex)] for from_vertex in from_]
    return vars

#Outgoing edge at Depot
for k in range(K):
    cqm.add_constraint(sum(outgoing_variables(0, k)) == 1, f'outgoing_edge_0_{k}')

#Each vertex should be visited at most once by a vehicle
for v in range(n):
    for k in range(K):
        cqm.add_constraint(sum(incoming_variables(v, k)) <= 1, f'incoming_veh_edge_{v}_{k}')

#Flow conservation
for v in range(n):
    for k in range(K):
        cqm.add_constraint(sum(incoming_variables(v, k)) -  sum(outgoing_variables(v, k)) == 0, f'flow_conserve_{v}_{k}')

#Each vertex should be covered at least once
for v in range(n):
    cqm.add_constraint(sum(incoming_variables(v)) >= 1, f'atleast_vehicle_visit_{v}')

#Sub-tour elimination
u = {}
for k in range(K):
    for i in range(n):
        u[(k, i)] = Integer(f'u_{k}_{i}', upper_bound=n)
 
Q, q = 10, 1 #max path length possible, assume 10 for now

for k in range(K):
    for i in range(n):
        for j in range(n):
            if W_adj[i][j] != 0 and j != 0:
                cqm.add_constraint(u[(k, i)] - u[(k, j)] + Q * variables[(k, i, j)] <= Q - q)

print(cqm)
print('-------------CQM--------------')
st = time.time()
sampler = LeapHybridCQMSampler()     
sampleset = sampler.sample_cqm(cqm,
                               time_limit=60,
                               label="VRP-multi visitation")  
print("{} feasible solutions of {}.".format(
      sampleset.record.is_feasible.sum(), len(sampleset)))   
et = time.time()
print(f'Exec Time: {et-st}')
print("-----------------Output-----------------")
# print(sampleset)

best = sampleset.filter(lambda row: row.is_feasible).first
print(best)

for key1, value1 in best.sample:
    if value1 == 1:
        print(key1, value1)
