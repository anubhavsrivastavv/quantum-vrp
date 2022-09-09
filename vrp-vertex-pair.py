import numpy as np
from collections import defaultdict 
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_graph(G):
    nx.draw_networkx(G)
    plt.show()


A = 10000
A_prime = 100

# 2 cycles
# n, K = 5, 2
# X = np.zeros(n*(n-1)).reshape(n*(n-1), 1)
# W_adj = np.array([[0, 10, 20], [5, 0, 25], [15, 0, 0]])
# W_adj = np.array([[0, 2, 0, 3], [0, 0, 5, 0], [4, 0, 0, 0], [5, 0, 0, 0]])
# W_adj = np.array([[0, 10, 0, 5, 12], [0, 0, 5, 0, 0], [0, 0, 0, 8, 0], [4, 0, 0, 0, 6], [8, 0, 0, 0, 0]])
# W_adj = np.array([[0, 5, 10, 0, 0], [6, 0, 0, 7, 0], [0, 0, 0, 0, 8], [6, 0, 0, 0, 0], [7, 0, 0, 9, 0]])
# alpha, beta = 1, 3

# 3 cycles
n, K = 6, 3
W_adj = np.array([[0, 10, 0, 5, 0, 12], [0, 0, 5, 0, 0, 0], [5, 0, 0, 0, 0, 8], [0, 0, 0, 0, 6, 0], [8, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]])
alpha, beta = 1, 2

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


#Edge-Vehicle combination
B, W_kij = [], []
for i in range(num_edges_in_graph):
    for b in range(K):
        B.append(str(b)+","+index_list[i])      #Therefore B just holds 'k,i,j'
        W_kij.append(W[i])         
        #print(B, W_kij)

num_edge_vehicle_combinations = len(W_kij)

Q = np.zeros(num_edge_vehicle_combinations*num_edge_vehicle_combinations).reshape(num_edge_vehicle_combinations, num_edge_vehicle_combinations)

for row_index in range(num_edge_vehicle_combinations):
    #To obtain indexes k,i,j back from B
    int_list = [int(r) for r in B[row_index].split(",")]
    k, i, j = int_list[0], int_list[1], int_list[2]
    
    for col_index in range(row_index, num_edge_vehicle_combinations):
        #To obtain indexes k,i,j back from B
        int_list2 = [int(r) for r in B[col_index].split(",")]
        k_prime, l , m = int_list2[0], int_list2[1], int_list2[2]

        #Diagonal 
        if row_index == col_index:
            Q[row_index][col_index] = W_kij[row_index]

            if i != 0:
                Q[row_index][col_index] += -A

            if j != 0:
                Q[row_index][col_index] += -A
            
            if i == 0:
                Q[row_index][col_index] += A*(1-2*K)
            
            if j == 0:
                Q[row_index][col_index] += A*(1-2*K)
            
            
        else:

            if i == l:
                Q[row_index][col_index] += 2*A
            
            if j == m:
                Q[row_index][col_index] += 2*A

        #For condition 6:
        if k != k_prime and i == alpha and l == beta:
            Q[row_index][col_index] += A

        #For condition 7:
        if i == l and j == m and k != k_prime:
            Q[row_index][col_index] += A

            

print()         
print('-----------------qubo matrix----------------')
print(Q)
print()

def index_to_x_kij(index):
    X_kij = B[index]
    k, i, j = X_kij.split(",")
    #print(f"index to Xij for : {index} is :{'x'+str(i)+str(j)}")
    return 'x'+str(k)+str(i)+str(j)

def convert_to_dict(qubo_matrix):
    qubo = defaultdict(float)
    for i in range(num_edge_vehicle_combinations):
        for j in range(num_edge_vehicle_combinations):
            if qubo_matrix[i][j] != 0:
                qubo[(index_to_x_kij(i), index_to_x_kij(j))] = qubo_matrix[i][j]
    print(qubo)
    return qubo

qubo = convert_to_dict(Q)
print(f'QUBO dict is: {qubo}')
# sys.exit()

# Direct QPU access
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample_qubo(qubo, num_reads = 100)

# # Hybrid Sampler access
# sampler = LeapHybridSampler()
# sampleset = sampler.sample_qubo(qubo)

print()
print("-----------------Output-----------------")
print(sampleset)
best_sol = sampleset.first.sample
print(best_sol)

sol_adj_matrix = np.zeros(shape = (n, n)) 
for (key, value) in best_sol.items():
    src, dst =key[2], key[3]
    if value == 1:
        sol_adj_matrix[int(src)][int(dst)] = 1

#NetworkX graph creation and plotting of solution
#G_base = nx.from_numpy_matrix(np.asmatrix(W_adj))
labels = [i for i in range(n)]
A2 = pd.DataFrame(sol_adj_matrix, index=labels, columns=labels)
G_sol = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())
plot_graph(G_sol)


                






