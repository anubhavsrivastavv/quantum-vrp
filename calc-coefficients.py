#X_ijk: is 1 if the edge i->j is taken by vehicle k else 0
K = 3
v = 10  #No. of nodes
def exists(i, j):
    #check if edge exists from i to j
    return True
for k in range(K):
    sum_X_ijk = 0
    for i in range(v):  #i is the current node, therefore this is for outgoing edge (i->j)
        for j in range(v):
            if exists(i, j):
                pass

def expand_summation(i, j, k):      #i and j are decided by v
    
