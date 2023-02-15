import numpy as np
from collections import defaultdict 
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler, LazyFixedEmbeddingComposite

from dwave_qbsolv import QBSolv
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler
from greedy import SteepestDescentSolver

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import sys
import itertools
import dimod
import collections
import dwave.inspector

def print_dict(d):
    for key, value in d.items():
        print(key,' : ',value)

def plot_graph(G, color_map):
    pos=nx.spring_layout(G) 
    nx.draw_networkx(G, pos, node_color= 'white')
    labels = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, label_pos=0.33)
    plt.show()


# A, B_penalty, C_penalty, D_penalty = 10000, 10000, 10000, 500   # 30 for k=3, 19.9999999999999999999 for k=2
A, B_penalty, C_penalty, D_penalty = 1000, 1000, 1000, 250 
A_prime = 100

# 2 cycles
# n, K, R = 5, 2, 2 
# X = np.zeros(n*(n-1)).reshape(n*(n-1), 1)
# W_adj = np.array([[0, 10, 20], [5, 0, 25], [15, 0, 0]])
W_adj = np.array([[0, 2, 0, 0], [0, 0, 5, 0], [5, 2, 0, 5], [10, 0, 0, 0]])         # 4node working
W_adj = np.array([[0, 2, 10, 15], [5, 0, 2, 0], [10, 0, 0, 2], [5, 0, 0, 0]])         # To test min cost paths are picked if r is visitation limit
W_adj = np.array([[0, 2, 10, 15], [5, 0, 2, 0], [10, 1, 0, 2], [10, 0, 0, 0]])
W_adj = np.array([[0, 8, 40, 60], [60, 0, 8, 0], [60, 4, 0, 8], [40, 0, 0, 0]])

#Checking c1-c3-c4
W_adj = np.array([[0, 30, 40, 60], [60, 0, 30, 0], [0, 24, 0, 30], [40, 0, 0, 0]])
W_adj = np.array([[0, 30, 40, 10], [60, 0, 30, 0], [0, 24, 0, 30], [10, 0, 0, 0]])  #shoould give 0-1-2-3-0, 0-3-0 and 0-3-0
# W_adj = np.array([[0, 5, 50, 10], [60, 0, 5, 0], [0, 24, 0, 5], [10, 0, 0, 0]])

#checking c1-c3
# W_adj = np.array([[0, 5, 50, 10], [60, 0, 5, 0], [0, 24, 0, 5], [10, 0, 0, 0]]) #should give 0-3-0
# W_adj = np.array([[0, 5, 50, 100], [60, 0, 5, 0], [0, 24, 0, 5], [10, 0, 0, 0]]) #should give 0-1-2-3-0
# W_adj = np.array([[0, 5, 50, 5], [60, 0, 5, 0], [5, 24, 0, 5], [10, 0, 0, 0]])  #should give 0-1-2-0 and/or 0-3-0

#To check the first constraint
# W_adj = np.array([[0, 5, 0], [5, 0, 10], [0, 0, 0]])
# n, K = 3, 4

# W_adj = np.array([[0, 10, 0, 5, 12], [0, 0, 5, 0, 0], [0, 0, 0, 8, 0], [4, 0, 0, 0, 6], [8, 0, 0, 0, 0]])
# W_adj = np.array([[0, 5, 10, 0, 0], [6, 0, 0, 7, 0], [0, 0, 0, 0, 8], [6, 0, 0, 0, 0], [7, 0, 0, 9, 0]])
# alpha, beta = 1, 3

# W_adj = np.array([[0, 5, 0, 0, 5], [0, 0, 4, 0, 0], [3, 0, 0, 2, 0], [4, 0, 0, 0, 0], [1, 0, 10, 0, 0]])
# W_adj = np.array([[0, 5, 5], [5, 0, 20], [5, 0, 0]])
# n, K, R = 3, 2, 2 
n, K, R = 4, 3, 3
color_map = ['white' for i in range(n)]
A, B_penalty, C_penalty, D_penalty = 1000, 1000, 1000, 1000

# # 3 cycles
# n, K, R = 6, 3, 3
# W_adj = np.array([[0, 10, 0, 5, 0, 12], [0, 0, 5, 0, 0, 0], [5, 0, 0, 0, 0, 8], [0, 0, 0, 0, 6, 0], [8, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]])
# alpha, beta = 1, 2

#Visualize the graph
labels = [i for i in range(len(W_adj))]     #To name the rows and cols of the pandas dataframe
A2 = pd.DataFrame(W_adj, index=labels, columns=labels)

G_base = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())
plot_graph(G_base, color_map)


"""
    This function returns a list of all edges in the graph in the (index_list) variable 
    and their corresponding weights in the (W) variable i.e, W[i] is the weight of the edge
    corresponding to the index_list[i] edge.
"""
def compute_W(W_adj):
    index_list, W =[], []
    k=0
    for u in range(n):
        for v in range(n):
            # print('k = ', k)
            if u == v or W_adj[u][v] == 0:
                continue
            else:
                index_list.append(str(u)+","+str(v))
                W.append(W_adj[u][v])
                #W[k] = W_adj[i][j]
                #k += 1
    return W, index_list


W, index_list= compute_W(W_adj)
num_edges_in_graph = len(W)     #No. of edges in the graph

print(W, "   ", index_list)

Q_Dict = {}
Q_Dict_Track = {}
print('-----Initial Fill-------')
for k1, k2 in itertools.product(range(K), range(K)):
    for i, j in itertools.product(range(n), range(n)):
            for l, m in itertools.product(range(n), range(n)):
                if W_adj[i][j] != 0  and W_adj[l][m] != 0:
                    print('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m), ':', 0)
                    Q_Dict[('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m))] = 0
                    Q_Dict_Track[('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m))] = '+0'
                    
                    if k1 == k2 and i == l and j == m:  #Linear terms
                        print('Linear terms')
                        print('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m), ':', W_adj[i][j])
                        Q_Dict[('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m))] += W_adj[i][j]
                        Q_Dict_Track[('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m))] += '+'+str(W_adj[i][j])
print('-----Initial Fill End-------')
# print('Q_Dict: ', Q_Dict)

# Q_Dict[('x001', 'x001')] += 10
# Q_Dict[('x001', 'x001')] += 20
# print('Accessing element', Q_Dict[('x001', 'x001')])     
# Q_Dict[('x001', 'x001')] = 0

                
def outgoing_edges(vertex):
    return [j for j in range(len(W_adj[vertex])) if W_adj[vertex][j] != 0]

def incoming_edges(vertex):
    return [i for i in range(len(W_adj)) if W_adj[i][vertex] != 0]

offset = 0

#Constraint 1
for k in range(K):
    for j in outgoing_edges(0):
        for j_prime in outgoing_edges(0):
            if j == j_prime:    #Linear Terms
                Q_Dict[('x'+str(k)+str(0)+str(j), 'x'+str(k)+str(0)+str(j_prime))] += -A
                Q_Dict_Track[('x'+str(k)+str(0)+str(j), 'x'+str(k)+str(0)+str(j_prime))] += str(-A)+' (outgoing edge(0) linear terms)'
            else:   #All combination of outgoing edges
                Q_Dict[('x'+str(k)+str(0)+str(j), 'x'+str(k)+str(0)+str(j_prime))] += 2*A
                Q_Dict_Track[('x'+str(k)+str(0)+str(j), 'x'+str(k)+str(0)+str(j_prime))] += '+'+str(2*A)+' (outgoing edge(0) all combination)'
    offset += A

# for k in range(K):
#      for i in outgoing_edges(1):
#         Q_Dict[('x'+str(k)+str(1)+str(i), 'x'+str(k)+str(1)+str(i))] += A

#Constraint 2
for v in range(1, n):
    for k in range(K):
        for i in incoming_edges(v):
            for i_prime in incoming_edges(v):
                if i != i_prime:
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))] += 2*B_penalty
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))] += '+'+str(2*B_penalty)+' (constraint-2 incoming edge('+str(v)+') all combination)' 

#Constraint 3
for v in range(0, n):
    for k in range(K):
        for i in incoming_edges(v):
            for i_prime in incoming_edges(v):
                if i == i_prime:
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i)+str(v))] += C_penalty
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i)+str(v))] += '+'+str(C_penalty)+' (constraint-3 incoming edge('+str(v)+') linear terms)' 
                else:
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))] += 2*C_penalty
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))] += '+'+str(2*C_penalty)+' (constraint-3 incoming edge('+str(v)+') all combination)' 

        for j in outgoing_edges(v):
            for j_prime in outgoing_edges(v):
                if j == j_prime:
                    Q_Dict[('x'+str(k)+str(v)+str(j), 'x'+str(k)+str(v)+str(j_prime))] += C_penalty
                    Q_Dict_Track[('x'+str(k)+str(v)+str(j), 'x'+str(k)+str(v)+str(j_prime))] += '+'+str(C_penalty)+' (constraint-3 outgoing edge('+str(v)+') linear terms)' 
                else:
                    Q_Dict[('x'+str(k)+str(v)+str(j), 'x'+str(k)+str(v)+str(j_prime))] += 2*C_penalty
                    Q_Dict_Track[('x'+str(k)+str(v)+str(j), 'x'+str(k)+str(v)+str(j_prime))] += '+'+str(2*C_penalty)+' (constraint-3 outgoing edge('+str(v)+') all combination)' 

        for i in incoming_edges(v):
            for j in outgoing_edges(v):
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(v)+str(j))] += -2*C_penalty
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(v)+str(j))] += str(-2*C_penalty)+' (constraint-3 one incoming and one outgoing edge('+str(v)+') all combination)' 

# Constraint 4
for v in range(1, n):
    for k, k_prime in itertools.product(range(K), range(K)):
        for i in incoming_edges(v):
            for i_prime in incoming_edges(v):
                if i == i_prime and k == k_prime: #Linear Terms
                    print('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))] -= R*D_penalty
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))] += str(-R*D_penalty)+' (constraint-4 incoming edge('+str(v)+') linear terms for all vehicles)' 
                else:
                    print('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(v))
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(v))] += 2*D_penalty
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(v))] += '+'+str(2*D_penalty)+' (constraint-4 incoming edge('+str(v)+') all combination terms for all vehicles)' 
    offset += R*D_penalty

# for v in range(1, n):
#     for k in range(K):
#         for i in incoming_edges(v):
#             print('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i)+str(v))
#             Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i)+str(v))] = -D_penalty
#             Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i)+str(v))] += str(-R*D_penalty)+' (constraint-4 incoming edge('+str(v)+') linear terms for all vehicles)' 
#     offset += R*D_penalty

# for v in range(0, n):
#     for k in range(K):
#         for i in incoming_edges(v):   
#             # print('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))
#             Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i)+str(v))] -= D_penalty/K
#             Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i)+str(v))] += str(-R*D_penalty)+' (constraint-4 incoming edge('+str(v)+') linear terms for all vehicles)'                
#     offset += D_penalty

# for v, v_prime in itertools.product(range(n), range(n)):
#     # print('Hello')
#     for k, k_prime in itertools.product(range(K), range(K)):
#         print(v, v_prime, k, k_prime)
#         for i in incoming_edges(v):
#             for i_prime in incoming_edges(v_prime):
#                 Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(v_prime))] -= D_penalty/n**2
#                 Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(v_prime))] += '-'+str(D_penalty)+' (constraint-4 incoming edge combinations for vertex pair('+str(v)+', '+str(v_prime)+') ' 
                    
# for v in range(1, n):
#     # print('Hello')
#     for k, k_prime in itertools.product(range(K), range(K)):
#         print(v, 0, k, k_prime)
#         for i in incoming_edges(v):
#             for i_prime in incoming_edges(0):
#                 Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(0))] -= D_penalty/len(incoming_edges(0))
#                 Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(0))] += '-'+str(D_penalty)+' (constraint-4 incoming edge combinations for vertex pair('+str(v)+', '+str(0)+') ' 

                    

print()         
print('-----------------qubo matrix----------------')
# print(Q)
print()

# def index_to_x_kij(index):
#     X_kij = B[index]
#     k, i, j = X_kij.split(",")
#     #print(f"index to Xij for : {index} is :{'x'+str(i)+str(j)}")
#     return 'x'+str(k)+str(i)+str(j)

# def convert_to_dict(qubo_matrix):
#     qubo = defaultdict(float)
#     for i in range(num_edge_vehicle_combinations):
#         for j in range(num_edge_vehicle_combinations):
#             if qubo_matrix[i][j] != 0:
#                 qubo[(index_to_x_kij(i), index_to_x_kij(j))] = qubo_matrix[i][j]
#     print(qubo)
#     return qubo

# qubo = convert_to_dict(Q)
# print(f'QUBO dict is: {qubo}')
# sys.exit()

# print(f'QUBO dict is: {Q_Dict}')
print_dict(Q_Dict)
# print(f'Tracking QUBO dict is:')
# print_dict(Q_Dict_Track)

def print_json(dictionary):
    """
    Create json from the Dict
    """
    json = '{'
    prev_key = ''
    count = 0

    #Sort the keys of dict and store it in list
    keys = dictionary.keys()
    sorted_keys = sorted(keys, key=lambda tup: tup[0])
    # print(sorted_keys)
    for key in sorted_keys:
        if key[0] != prev_key:
            if count != 0:
                json += '}, '
            json += key[0]+': {'+key[1]+': "'+str(dictionary[key])+'"'
            prev_key = key[0]
        else:
            json += ', '+key[1]+': "'+str(dictionary[key])+'"'
        count += 1
    json += '}}'
    print('json: ', json)

def print_json_track(Q_Dict_Track, Q_Dict):
    """
    Create json from the Dict
    """
    json = '{'
    prev_key = ''
    count = 0

    #Sort the keys of dict and store it in list
    keys = Q_Dict_Track.keys()
    sorted_keys = sorted(keys, key=lambda tup: tup[0])
    # print(sorted_keys)
    for key in sorted_keys:
        if key[0] != prev_key:
            if count != 0:
                json += '}, '
            json += key[0]+': {'+key[1]+': "'+str(Q_Dict[key])+'\\n'+str(Q_Dict_Track[key])+'"'
            prev_key = key[0]
        else:
            json += ', '+key[1]+': "'+str(Q_Dict[key])+'\\n'+str(Q_Dict_Track[key])+'"'
        count += 1
    json += '}}'
    print('json: ', json)

print_json(Q_Dict)
print_json_track(Q_Dict_Track, Q_Dict)

qubo = defaultdict(float, Q_Dict)

print('-------------BQM--------------')
bqm = dimod.BQM(dimod.vartypes.Vartype.BINARY)
for key, value in qubo.items():
    e1, e2 = key[0], key[1]          
    # print(e1, ': ', e2 ,' = ', value)
    if e1 == e2:
        bqm.add_linear(e1, value)
    else:
        bqm.add_quadratic(e1, e2, value)

# offset = 0
# bqm.add_offset(offset)

# print(bqm.adj)
# print('Offset: ', offset)
print('-------------BQM Construction End--------------')


# print('-------------')
# for e1 in ['x001', 'x010', 'x002', 'x020', 'x101', 'x110', 'x102', 'x120']:
#     for e2 in ['x001', 'x010', 'x002', 'x020', 'x101', 'x110', 'x102', 'x120']:
#         if Q_Dict[(e1, e2)] != 0:
#                 print((e1, e2), ':', Q_Dict[(e1, e2)])

# Direct QPU access
# sampler = EmbeddingComposite(DWaveSampler())

# sampler = LazyFixedEmbeddingComposite(DWaveSampler())
# sampler = QBSolv()
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(qubo, num_reads = 1000)
# sampleset = sampler.sample_qubo(qubo, num_reads = 1000, chain_strength = 2000)
# sampleset = sampler.sample(bqm, num_reads = 1000)

# # Hybrid Sampler access
# sampler = LeapHybridSampler()
# sampleset = sampler.sample_qubo(qubo)

print()
print("-----------------Output-----------------")
print(sampleset)

best_sol = sampleset.first.sample
print(best_sol)
# print(sampleset.lowest().record)

sol_adj_matrix = np.zeros(shape = (n, n)) 
for (key, value) in best_sol.items():
    src, dst =key[2], key[3]
    if value == 1:
        sol_adj_matrix[int(src)][int(dst)] = W_adj[int(src)][int(dst)]

#NetworkX graph creation and plotting of solution
#G_base = nx.from_numpy_matrix(np.asmatrix(W_adj))
labels = [i for i in range(n)]
# A2 = pd.DataFrame(sol_adj_matrix, index=labels, columns=labels)
# G_sol = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())
G_sol = nx.from_numpy_matrix(sol_adj_matrix, create_using=nx.DiGraph())
plot_graph(G_sol, color_map)
count = 0
expected_edge_keys = ['03', '01', '12', '23', '30']
# expected_sol = {'x001': 1, 'x002': 0, 'x003': 0, 'x010': 1, 'x012': 0, 'x020': 0, 'x021': 0, 'x023': 0, 'x030': 0, 'x101': 1, 'x102': 0, 'x103': 0, 'x110': 1, 'x112': 0, 'x120': 0, 'x121': 0, 'x123': 0, 'x130': 0, 'x201': 1, 'x202': 0, 'x203': 0, 'x210': 0, 'x212': 1, 'x220': 0, 'x221': 0, 'x223': 1, 'x230': 1}
# for datum in sampleset.data(fields=['sample', 'energy']):
#     count += 1
#     # if datum.energy == sampleset.first.energy:
#     # if  count < 150:
#     sol = datum.sample
#     flag = True
#     # for key in expected_sol.keys():
#     #     print(expected_sol[key], sol[key], end = ' ')
#     #     if expected_sol[key] != sol[key]:
#     #         flag = False
#     # print('\n')
#     # if flag:
#     #     print('Found sol:   ', sol, ' ', datum.energy)
#     print(sol, datum.energy)
#     if count >= 10:
#         break
for datum in sampleset.data(fields=['sample', 'energy']):
    sol = datum.sample
    
    match = 0
    for edge in expected_edge_keys:
        flag_edge_picked = False
        for veh_edge, value in sol.items():
            if veh_edge[2:] == edge and value == 1:
                #edge picked up
                # print('edge: ', edge, 'veh_edge: ', veh_edge)
                flag_edge_picked = True
                break
        if flag_edge_picked:
            match += 1
    
    # if match >= 4 and (sol['x010'] != 1 and sol['x110'] != 1 and sol['x210'] != 1):
    #     print('matched: ', sol, datum.energy)
    # else:
    #     # print('not matched')
    #     pass

# print(Q_Dict)
# print(Q_Dict_Track)

def sol_explainer(sol, energy):
    print()
    print('------------------------------')
    print('Explanation for the solution: ')
    penalties, energy_computed = '', 0
    for key1, value1 in sol.items():
        for key2, value2 in sol.items():
            if value1 == 1 and value2 == 1 and Q_Dict[(key1, key2)] != 0:
                print(f'For {(key1, key2)} : {Q_Dict[(key1, key2)]} which is because of : {Q_Dict_Track[(key1, key2)]}')
                penalties += f'({Q_Dict[(key1, key2)]})+'
                energy_computed += Q_Dict[(key1, key2)]
    print()
    print(f'total_reported = {energy} , energy_computed = {energy_computed} because ', penalties)
    print('------------------------------')
    print()

sol_explainer(best_sol, -8890.0)

expected_sol = {'x001': 0, 'x002': 0, 'x003': 1, 'x010': 0, 'x012': 0, 'x021': 0, 'x023': 0, 'x030': 1, 'x101': 1, 'x102': 0, 'x103': 0, 'x110': 0, 'x112': 1, 'x121': 0, 'x123': 0, 'x130': 1, 'x201': 0, 'x202': 0, 'x203': 1, 'x210': 0, 'x212': 0, 'x221': 0, 'x223': 0, 'x230': 1} 
sol_explainer(expected_sol, -8890.0)
                

dwave.inspector.show(sampleset)
print('chain-strength: ', sampleset.info['embedding_context']['chain_strength'])