#Import libraries
import pandas as pd
import itertools 
import numpy as np
import random
import networkx as nx


import matplotlib as mpl
import matplotlib.pyplot as plt

from docplex.mp.model import Model
import docplex.mp.vartype as vartype
# from docplex.mp.basic import ObjectiveSense

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


m = Model(name='vrp',log_output=False)
m.objective_sense = 'min'
m.parameters.threads.set(1)

####VARIABLES####
variable_matrix  = [[[None for j in range(n)] for i in range(n)] for k in range(K)]
variables = m.binary_var_list([(k, i, j) for k in range(K) for i in range(n) for j in range(n) if W_adj[i][j] != 0], name = 'travel_on')
cost = [W_adj[i][j] for k in range(K) for i in range(n) for j in range(n) if W_adj[i][j] != 0]
for k in range(K):
    edges = 0
    for i in range(n):
        for j in range(n):
            if W_adj[i][j] != 0:
                variable_matrix[k][i][j] = variables[k*e+edges]
                edges += 1


# m.binary_var_list([(row.point_from, row.point_to) for row in data.itertuples()], name = 'travel_on')

####OBJECTIVE####
total_cost = m.sum([cost[i]*variables[i] for i in range(len(variables))])
# m.add_kpi(total_cost, "distance")
m.minimize(total_cost)

####CONSTRAINTS####
def outgoing_variables(vertex, k = None):
    to = [j for j in range(len(W_adj[vertex])) if W_adj[vertex][j] != 0]
    if k is None:
        variables = [variable_matrix[k][vertex][to_vertex] for k in range(K) for to_vertex in to]
    else:
        variables = [variable_matrix[k][vertex][to_vertex] for to_vertex in to]
    return variables

def incoming_variables(vertex, k = None):
    from_ = [i for i in range(len(W_adj)) if W_adj[i][vertex] != 0]
    if k is None:
        variables = [variable_matrix[k][from_vertex][vertex] for k in range(K) for from_vertex in from_]
    else:
        variables = [variable_matrix[k][from_vertex][vertex] for from_vertex in from_]
    return variables

#Outgoing edge at Depot
# out_0 = outgoing_variables(0)
for k in range(K):
    m.add_constraint(m.sum(outgoing_variables(0, k)) == 1, 'outgoing_edge_0_')

#Each vertex should be visited at most once by a vehicle
for v in range(n):
    for k in range(K):
        m.add_constraint(m.sum(incoming_variables(v, k)) <= 1, 'incoming_veh_edge_'+str(v))

#Flow conservation
for v in range(n):
    for k in range(K):
        m.add_constraint(m.sum(incoming_variables(v, k)) -  m.sum(outgoing_variables(v, k)) == 0, 'flow_conserve_'+str(v))

#Each vertex should be covered at least once
for v in range(n):
    m.add_constraint(m.sum(incoming_variables(v)) >= 1, 'atleast_vehicle_visit_'+str(v))


#Sub-tour elimination
u = m.integer_var_dict([(k, i) for k in range(K) for i in range(n)], name='u')
Q, q = 10, 1 #max path length possible, assume 10 for now

for k in range(K):
    for i in range(n):
        for j in range(n):
            if W_adj[i][j] != 0 and j != 0:
                m.add_constraint(u[(k, i)] - u[(k, j)] + Q * variable_matrix[k][i][j] <= Q - q)



import json, time
st = time.time()
solution = m.solve()
et = time.time()
print(f'Exec Time: {et-st}')
sol_json = solution.export_as_json_string()
print('----------Problem Details: -----------')
print(sol_json)
print('----------Solution Details: -----------')
print(json.loads(sol_json)['CPLEXSolution']['variables'])
variable_values = [var.solution_value for var in variables]
print(variable_values)

op_dict = {}
def parse_output():
    for k in range(K):
        edges = 0
        for i in range(n):
            for j in range(n):
                if W_adj[i][j] != 0:
                    op_dict['x_'+str(k)+str(i)+str(j)] = variable_values[k*e+edges]
                    edges += 1
    print()
    print(op_dict)

parse_output()
print()
print(f'n: {n}, K: {K}')
print([key for key, val in op_dict.items() if val == 1])
# print(m.kpis_as_dict()['distance'])
print(np.dot(variable_values, cost))

