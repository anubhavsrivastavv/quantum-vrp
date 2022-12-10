import numpy as np
from collections import defaultdict 
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import sys
import itertools
import dimod

def print_dict(d):
    for key, value in d.items():
        print(key,' : ',value)

def plot_graph(G):
    pos=nx.spring_layout(G) 
    nx.draw_networkx(G,pos)
    labels = nx.get_edge_attributes(G,'weight')
    # nx.draw_networkx(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


A, B_penalty, C_penalty, D_penalty = 10000, 10000, 10000, 1000   # 30 for k=3, 19.9999999999999999999 for k=2
A_prime = 100

# 2 cycles
# n, K, R = 5, 2, 2 
# X = np.zeros(n*(n-1)).reshape(n*(n-1), 1)
# W_adj = np.array([[0, 10, 20], [5, 0, 25], [15, 0, 0]])
W_adj = np.array([[0, 2, 0, 0], [0, 0, 5, 0], [5, 2, 0, 5], [10, 0, 0, 0]])         # 4node working
W_adj = np.array([[0, 2, 10, 15], [5, 0, 2, 0], [10, 0, 0, 2], [5, 0, 0, 0]])         # To test min cost paths are picked if r is visitation limit
W_adj = np.array([[0, 2, 10, 15], [5, 0, 2, 0], [10, 1, 0, 2], [10, 0, 0, 0]])
# W_adj = np.array([[0, 10, 0, 5, 12], [0, 0, 5, 0, 0], [0, 0, 0, 8, 0], [4, 0, 0, 0, 6], [8, 0, 0, 0, 0]])
# W_adj = np.array([[0, 5, 10, 0, 0], [6, 0, 0, 7, 0], [0, 0, 0, 0, 8], [6, 0, 0, 0, 0], [7, 0, 0, 9, 0]])
# alpha, beta = 1, 3

# W_adj = np.array([[0, 5, 0, 0, 5], [0, 0, 4, 0, 0], [3, 0, 0, 2, 0], [4, 0, 0, 0, 0], [1, 0, 10, 0, 0]])
# W_adj = np.array([[0, 5, 5], [5, 0, 20], [5, 0, 0]])
# n, K, R = 3, 2, 2 
n, K, R = 4, 3, 4

# # 3 cycles
# n, K, R = 6, 3, 3
# W_adj = np.array([[0, 10, 0, 5, 0, 12], [0, 0, 5, 0, 0, 0], [5, 0, 0, 0, 0, 8], [0, 0, 0, 0, 6, 0], [8, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]])
# alpha, beta = 1, 2

#Visualize the graph
labels = [i for i in range(len(W_adj))]     #To name the rows and cols of the pandas dataframe
A2 = pd.DataFrame(W_adj, index=labels, columns=labels)

G_base = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())
plot_graph(G_base)


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


# #Edge-Vehicle combination
# B, W_kij = [], []
# for i in range(num_edges_in_graph):
#     for b in range(K):
#         B.append(str(b)+","+index_list[i])      #Therefore B just holds 'k,i,j'
#         W_kij.append(W[i])         
#         #print(B, W_kij)

# num_edge_vehicle_combinations = len(W_kij)

# Q = np.zeros(num_edge_vehicle_combinations*num_edge_vehicle_combinations).reshape(num_edge_vehicle_combinations, num_edge_vehicle_combinations)

# for row_index in range(num_edge_vehicle_combinations):
#     #To obtain indexes k,i,j back from B
#     int_list = [int(r) for r in B[row_index].split(",")]
#     k, i, j = int_list[0], int_list[1], int_list[2]
    
#     for col_index in range(row_index, num_edge_vehicle_combinations):
#         #To obtain indexes k,i,j back from B
#         int_list2 = [int(r) for r in B[col_index].split(",")]
#         k_prime, l , m = int_list2[0], int_list2[1], int_list2[2]

#         #Diagonal 
#         if row_index == col_index:
#             Q[row_index][col_index] = W_kij[row_index]

#             #Constraint 1:  Each vehicle should exit 0 exactly once
#             if i == 0:  
#                 Q[row_index][col_index] += -A

#             #Constraint 3:  Number of times a vehicle enters enters is the number of times it exits:
#             Q[row_index][col_index] += 2*A #(1 as an incoming edge and 1 as an outgoing edge)

#             #Constraint 4: Each vertex must be visited atleast 1 and atmost r times
#             #Each linear term has coefficient 1
#             Q[row_index][col_index] += -R*B_penalty

#             # if i != 0:
#             #     Q[row_index][col_index] += -A

#             # if j != 0:
#             #     Q[row_index][col_index] += -A
            
#             # if i == 0:
#             #     Q[row_index][col_index] += A*(1-2*K)
            
#             # if j == 0:
#             #     Q[row_index][col_index] += A*(1-2*K)
            
            
#         else:

#             #For same vehicle
#             if k == k_prime:
                
#                 #Constraint 1:  Combination of outgoing edges of 0 - each vehicle should exit 0 exactly once
#                 if i == 0 and l == 0:
#                     Q[row_index][col_index] += 2*A

                
#                 #Constraint 2:  Incoming edges of same vertex - each vehicle can either visit a vertex once or not at all
#                 if j == m:
#                     Q[row_index][col_index] += 2*A

                
#                 #Constraint 3:  Number of times a vehicle enters enters is the number of times it exits:
#                 #For combination of incoming edges at j (or m):
#                 if i != l and j == m:
#                     Q[row_index][col_index] += 2*A
#                 #For combination of outgoing edges at i (or l):
#                 if i == l and j != m:
#                     Q[row_index][col_index] += 2*A
#                 #For combination of each incoming and outgoing edge at vertex j (i->j) (l->m) where j == l:
#                 if j == l:
#                     Q[row_index][col_index] -= 2*A
            
            
#                 #Constraint 4: Each vertex must be visited atleast 1 and atmost r times
#                 #all combinations of vehicles but they must be incoming at same vertex
#                 if i != l and j == m:
#                     Q[row_index][col_index] += 2*B_penalty
            
#             else:

#                 #Constraint 4: Each vertex must be visited atleast 1 and atmost r times
#                 #all combinations of vehicles but they must be incoming at same vertex
#                 if j == m:
#                     Q[row_index][col_index] += 2*B_penalty


#             # if i == l:
#             #     Q[row_index][col_index] += 2*A
            
#             # if j == m:
#             #     Q[row_index][col_index] += 2*A

#         # #For condition 6:
#         # if k != k_prime and i == alpha and l == beta:
#         #     Q[row_index][col_index] += A

#         # #For condition 7:
#         # if i == l and j == m and k != k_prime:
#         #     Q[row_index][col_index] += A

Q_Dict = {}
Q_Dict_Track = {}
for k1, k2 in itertools.product(range(K), range(K)):
    for i, j in itertools.product(range(n), range(n)):
            for l, m in itertools.product(range(n), range(n)):
                if W_adj[i][j] != 0  and W_adj[l][m] != 0:
                    Q_Dict[('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m))] = 0
                    Q_Dict_Track[('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m))] = '+0'
                    
                    if k1 == k2 and i == l and j == m:  #Linear terms
                        Q_Dict[('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m))] += W_adj[i][j]
                        Q_Dict_Track[('x'+str(k1)+str(i)+str(j), 'x'+str(k2)+str(l)+str(m))] += '+'+str(W_adj[i][j])

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
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(v)+str(j))] -= 2*C_penalty
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(v)+str(j))] += str(-2*C_penalty)+' (constraint-3 one incoming and one outgoing edge('+str(v)+') all combination)' 

# Constraint 4
for v in range(0, n):
    for k, k_prime in itertools.product(range(K), range(K)):
        for i in incoming_edges(v):
            for i_prime in incoming_edges(v):
                if i == i_prime and k == k_prime: #Linear Terms
                    print('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))] -= R*D_penalty/K
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k)+str(i_prime)+str(v))] += str(-R*D_penalty)+' (constraint-4 incoming edge('+str(v)+') linear terms for all vehicles)' 
                else:
                    print('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(v))
                    Q_Dict[('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(v))] += 2*D_penalty/K
                    Q_Dict_Track[('x'+str(k)+str(i)+str(v), 'x'+str(k_prime)+str(i_prime)+str(v))] += '+'+str(2*D_penalty)+' (constraint-4 incoming edge('+str(v)+') all combination terms for all vehicles)' 
    offset += R*D_penalty

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

print(f'QUBO dict is: {Q_Dict}')
print_dict(Q_Dict)
print(f'Tracking QUBO dict is:')
print_dict(Q_Dict_Track)

qubo = defaultdict(float, Q_Dict)

print('-------------BQM--------------')
bqm = dimod.BQM(dimod.vartypes.Vartype.BINARY)
for key, value in qubo.items():
    e1, e2 = key[0], key[1]          
    print(e1, ': ', e2 ,' = ', value)
    if e1 == e2:
        bqm.add_linear(e1, value)
    else:
        bqm.add_quadratic(e1, e2, value)

# offset = 0
# bqm.add_offset(offset)

print(bqm.adj)
print('Offset: ', offset)
print('-------------BQM Construction End--------------')

# print('-------------')
# for e1 in ['x001', 'x010', 'x002', 'x020', 'x101', 'x110', 'x102', 'x120']:
#     for e2 in ['x001', 'x010', 'x002', 'x020', 'x101', 'x110', 'x102', 'x120']:
#         if Q_Dict[(e1, e2)] != 0:
#                 print((e1, e2), ':', Q_Dict[(e1, e2)])

# Direct QPU access
sampler = EmbeddingComposite(DWaveSampler())
# sampleset = sampler.sample_qubo(qubo, num_reads = 1000)
sampleset = sampler.sample(bqm, num_reads = 1000)

# # Hybrid Sampler access
# sampler = LeapHybridSampler()
# sampleset = sampler.sample_qubo(qubo)

print()
print("-----------------Output-----------------")
print(sampleset)

best_sol = sampleset.first.sample
print(best_sol)
print(sampleset.lowest().record)

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
plot_graph(G_sol)
