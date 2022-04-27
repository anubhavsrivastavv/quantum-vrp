import numpy as np
from collections import defaultdict 
from dwave.system import DWaveSampler, EmbeddingComposite

n, k = 3, 2
A = 20
X = np.zeros(n*(n-1)).reshape(n*(n-1), 1)
W_adj = np.array([[0, 2, 5], [3, 0, 4], [3, 6, 0]])

def compute_W(W_adj):
    W = np.zeros(shape = (n*(n-1), 1))
    
    k=0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            else:
                W[k] = W_adj[i][j]
                k += 1
    return W

W = compute_W(W_adj)
#print(f'W: {W}')

def compute_Zt(t):
    Zt = np.zeros(n*(n-1)).reshape(n*(n-1), 1)

    k=0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            else:
                if j == t:
                    Zt[k] = 1
                else:
                    Zt[k] = 0
                k += 1
    #print("Zt[0]: ", Zt)
    #print("shape is: ", np.shape(Zt))
    return np.transpose(Zt)

# Zt_0 = compute_Zt(0)
# Zt_1 = compute_Zt(1)
# Zt_2 = compute_Zt(2)

# print("Zt[0]: ", Zt_0, "shape is: ", np.shape(Zt_0))
# print("Zt[1]: ", Zt_1)
# print("Zt[2]: ", Zt_2)

def calc_Q():
    Zt_combined = np.zeros(shape = (n , n*(n-1)))
    for i in range(n):
        Zt_i = compute_Zt(i)
        Zt_combined[i] = Zt_i
    #print(f'Zt_combined:    {Zt_combined}')

    Z_total = np.matmul(np.transpose(Zt_combined), Zt_combined)
    #print(f'Z_total:    {Z_total}')
    
    I_n = np.identity(n)
    J_n_1 = np.ones(shape = (n-1, n-1))

    tensor_product = np.kron(I_n, J_n_1)
    #print(f'tensor_product:    {tensor_product}')

    Q = A*(Z_total + tensor_product)
    print(f'Q:    {Q}')
    return Q

def calc_G():
    Zt_0 = compute_Zt(0)
    J_n_1 = np.ones(shape = (n-1, 1))
    
    e_0 = np.zeros(shape = (n, 1))
    e_0[0] = 1

    part1 = np.kron(e_0, J_n_1) + np.transpose(Zt_0)
    
    J_n = np.ones(shape = (n, 1))
    part2 = np.kron(J_n, J_n_1)

    g = W - 2*A*k*part1 + 2*A*part2

    print(f'g: {g}')
    return g

def find_coefficients(Q, g):
    # for i in range(n):
    #     for j in range(n):
    #         if i == j:
    #             continue

    #Take sum of Q about diagonal
    for i in range(n*(n-1)):
        for j in range(i+1, n*(n-1)):
            Q[i][j] += Q[j][i]
            Q[j][i] = 0

    for i in range(n*(n-1)):
        Q[i][i] += g[i]

    print('Final QUBO matrix is:    ', Q)
    return Q

def index_to_x_ij(index):
    counter = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            else:
                if counter == index:
                    return 'x'+str(i)+str(j)
                counter += 1

def convert_to_dict(qubo_matrix):
    qubo = defaultdict(float)
    for i in range(n*(n-1)):
        for j in range(i, n*(n-1)):
            qubo[(index_to_x_ij(i), index_to_x_ij(j))] = qubo_matrix[i][j]
    print(qubo)
    return qubo



Q = calc_Q()
g = calc_G()
c = 2*A*(n-1) + 2*A*(k**2)
#print(f'c:  {c}')
qubo_matrix = find_coefficients(Q, g)
qubo = convert_to_dict(qubo_matrix)

sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample_qubo(qubo, num_reads = 100)
#sampleset = sampler.sample_qubo(qubo_matrix, num_reads = 10)

print()
print("-----------------Output-----------------")
print(sampleset)