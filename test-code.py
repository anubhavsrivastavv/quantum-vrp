import itertools
import numpy as np
K, n = 3, 4
W_adj = np.array([[0, 2, 10, 15], [50, 0, 2, 0], [50, 1, 0, 2], [10, 0, 0, 0]])
for k1, k2 in itertools.product(range(K), range(K)):
    for i, j in itertools.product(range(n), range(n)):
            for l, m in itertools.product(range(n), range(n)):
                if W_adj[i][j] != 0  and W_adj[l][m] != 0:
                    print('inside')