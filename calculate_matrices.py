import numpy as np

n, k = 3, 2
A = 100
X = np.zeros(n*(n-1)).reshape(n*(n-1), 1)
#print(X)

def compute_W(G):
    

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
    print(f'Zt_combined:    {Zt_combined}')

    Z_total = np.matmul(np.transpose(Zt_combined), Zt_combined)
    print(f'Z_total:    {Z_total}')
    
    I_n = np.identity(n)
    J_n_1 = np.ones(shape = (n-1, n-1))

    tensor_product = np.kron(I_n, J_n_1)
    print(f'tensor_product:    {tensor_product}')

    Q = A*(Z_total + tensor_product)
    print(f'Q:    {Q}')

def calc_G():
    Zt_0 = compute_Zt(0)
    J_n_1 = np.ones(shape = (n-1, 1))
    
    e_0 = np.zeros(shape = (n, 1))
    e_0[0] = 1

    part1 = np.kron(e_0, J_n_1) + np.transpose(Zt_0)
    
    J_n = np.ones(shape = (n, 1))
    part2 = np.kron(J_n, J_n_1)

    g = 


calc_Q()