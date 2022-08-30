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

# n, k = 3, 2
# n, k = 4, 2
# n, k = 4, 3
n, k = 5, 2


A = 1000
X = np.zeros(n*(n-1)).reshape(n*(n-1), 1)
# W_adj = np.array([[0, 10, 20], [5, 0, 25], [15, 0, 0]])
W_adj = np.array([[0, 10, 30, 0, 40], [10, 0, 30, 15, 0], [0, 0, 0, 20, 0], [0, 0, 0, 0, 30], [30, 0, 0, 0, 0]])
#W_adj = np.array([[0, 36.84, 5.06, 30.63], [36.84, 0, 24.55, 63.22], [5.06, 24.55, 0, 15.50], [30.63, 63.22, 15.50, 0]])
# W_adj = np.array([[0, 6.794, 61.653, 24.557, 47.767],[6.794, 0, 87.312, 47.262, 39.477],[61.653, 87.312, 0, 9.711, 42.887],[24.557, 47.262, 9.711, 0, 40.98],[47.767, 39.477, 42.887, 40.98, 0]])

#NetworkX graph creation and plotting
#G_base = nx.from_numpy_matrix(np.asmatrix(W_adj))
labels = [i for i in range(n)]
A2 = pd.DataFrame(W_adj, index=labels, columns=labels)
G_base = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())
plot_graph(G_base)

def compute_W(W_adj):
    #W = np.zeros(shape = (n*(n-1), 1))
    map, W =[], []
    k=0
    for i in range(n):
        for j in range(n):
            # print('k = ', k)
            if i == j:
                continue
            else:
                if W_adj[i][j] != 0:
                    map.append(str(i)+","+str(j))
                    W.append(W_adj[i][j])
                    #W[k] = W_adj[i][j]
                    #k += 1
    return W, map

W, index_map= compute_W(W_adj)

print(W, "   ", index_map)

Q = np.zeros(len(W)*len(W)).reshape(len(W), len(W))

for row_index in range(len(W)):
    int_list = [int(r) for r in index_map[row_index].split(",")]
    i , j = int_list[0], int_list[1]
    # print('type: ', type(i), type(j))
    for col_index in range(row_index,len(W)):
        int_list2 = [int(r) for r in index_map[col_index].split(",")]
        m , l = int_list2[0], int_list2[1]
        # print('(i,j), (k,l): ',(i,j), (k,l))

        #Diagonal 
        if row_index == col_index:
            Q[row_index][col_index] = W[row_index]
            #print(f'Inside1 and i:{i}, j:{j}, k:{k} and l:{l}')
            if i == 0 or l == 0:
                Q[row_index][col_index] += (-2*k+1) * A
                # print('Inside2')
                print(f' Q[{row_index}][{col_index}]:  {Q[row_index][col_index]}')
            # elif l == 0:
            #     Q[row_index][col_index] += (-2*k+1) * A
            if i != 0:
                Q[row_index][col_index] -= A
            if l != 0:
                Q[row_index][col_index] -= A
        else:
            #if (i != 0 and k != 0) and (i==j or k==l):
            if i==m or j==l:
                 Q[row_index][col_index] = 2*A  
print()         
print('-----------------qubo matrix----------------')
print(Q)
print()

def index_to_x_ij(index):
    X_ij = index_map[index]
    i, j = X_ij.split(",")
    #print(f"index to Xij for : {index} is :{'x'+str(i)+str(j)}")
    return 'x'+str(i)+str(j)

def convert_to_dict(qubo_matrix):
    qubo = defaultdict(float)
    for i in range(len(W)):
        for j in range(len(W)):
            if qubo_matrix[i][j] != 0:
                qubo[(index_to_x_ij(i), index_to_x_ij(j))] = qubo_matrix[i][j]
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
    src, dst =key[1], key[2]
    if value == 1:
        sol_adj_matrix[int(src)][int(dst)] = 1

#NetworkX graph creation and plotting of solution
#G_base = nx.from_numpy_matrix(np.asmatrix(W_adj))
labels = [i for i in range(n)]
A2 = pd.DataFrame(sol_adj_matrix, index=labels, columns=labels)
G_sol = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())
plot_graph(G_sol)



# #print(f'W: {W}')

# def compute_Zt(t):
#     Zt = np.zeros(n*(n-1)).reshape(n*(n-1), 1)

#     k=0
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             else:
#                 if j == t:
#                     Zt[k] = 1
#                 else:
#                     Zt[k] = 0
#                 k += 1
#     #print("Zt[0]: ", Zt)
#     #print("shape is: ", np.shape(Zt))
#     return np.transpose(Zt)

# # Zt_0 = compute_Zt(0)
# # Zt_1 = compute_Zt(1)
# # Zt_2 = compute_Zt(2)

# # print("Zt[0]: ", Zt_0, "shape is: ", np.shape(Zt_0))
# # print("Zt[1]: ", Zt_1)
# # print("Zt[2]: ", Zt_2)

# def calc_Q():
#     Zt_combined = np.zeros(shape = (n , n*(n-1)))
#     for i in range(n):
#         Zt_i = compute_Zt(i)
#         Zt_combined[i] = Zt_i
#     #print(f'Zt_combined:    {Zt_combined}')

#     Z_total = np.matmul(np.transpose(Zt_combined), Zt_combined)
#     #print(f'Z_total:    {Z_total}')
    
#     I_n = np.identity(n)
#     J_n_1 = np.ones(shape = (n-1, n-1))

#     tensor_product = np.kron(I_n, J_n_1)
#     #print(f'tensor_product:    {tensor_product}')

#     Q = A*(Z_total + tensor_product)
#     print(f'Q:    {Q}')
#     return Q

# def calc_G():
#     Zt_0 = compute_Zt(0)
#     J_n_1 = np.ones(shape = (n-1, 1))
    
#     e_0 = np.zeros(shape = (n, 1))
#     e_0[0] = 1

#     part1 = np.kron(e_0, J_n_1) + np.transpose(Zt_0)
    
#     J_n = np.ones(shape = (n, 1))
#     part2 = np.kron(J_n, J_n_1)

#     g = W + (2*A - 2*A*k)*part1 - 4*A*part2

#     print(f'g: {g}')
#     return g

# def find_coefficients(Q, g):
#     # for i in range(n):
#     #     for j in range(n):
#     #         if i == j:
#     #             continue

#     #Take sum of Q about diagonal
#     for i in range(n*(n-1)):
#         for j in range(i+1, n*(n-1)):
#             Q[i][j] += Q[j][i]
#             Q[j][i] = 0

#     for i in range(n*(n-1)):
#         Q[i][i] += g[i]

#     print('Final QUBO matrix is:    ', Q)
#     return Q

# def index_to_x_ij(index):
#     counter = 0
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             else:
#                 if counter == index:
#                     return 'x'+str(i)+str(j)
#                 counter += 1

# def convert_to_dict(qubo_matrix):
#     qubo = defaultdict(float)
#     for i in range(n*(n-1)):
#         for j in range(i, n*(n-1)):
#             qubo[(index_to_x_ij(i), index_to_x_ij(j))] = qubo_matrix[i][j]
#     print(qubo)
#     return qubo


# Q = calc_Q()
# g = calc_G()
# c = 2*A*(n-1) + 2*A*(k**2)
# #print(f'c:  {c}')
# qubo_matrix = find_coefficients(Q, g)
# qubo = convert_to_dict(qubo_matrix)

# # Direct QPU access
# sampler = EmbeddingComposite(DWaveSampler())
# sampleset = sampler.sample_qubo(qubo, num_reads = 100)

# # # Hybrid Sampler access
# # sampler = LeapHybridSampler()
# # sampleset = sampler.sample_qubo(qubo)

# print()
# print("-----------------Output-----------------")
# print(sampleset)
# best_sol = sampleset.first.sample
# print(best_sol)

# sol_adj_matrix = np.zeros(shape = (n, n)) 
# for (key, value) in best_sol.items():
#     src, dst =key[1], key[2]
#     if value == 1:
#         sol_adj_matrix[int(src)][int(dst)] = 1

# #NetworkX graph creation and plotting of solution
# #G_base = nx.from_numpy_matrix(np.asmatrix(W_adj))
# labels = [i for i in range(n)]
# A2 = pd.DataFrame(sol_adj_matrix, index=labels, columns=labels)
# G_sol = nx.from_pandas_adjacency(A2, create_using=nx.DiGraph())
# plot_graph(G_sol)
