#Import libraries
import pandas
import itertools 
import numpy
import random
import networkx


import matplotlib as mpl
import matplotlib.pyplot as plt

from docplex.mp.model import Model

#create the base model
def tsp(data, max_length):
    m = Model(name='tsp',log_output=False)
    m.parameters.threads.set(1)
    
    all_locations = set(data.point_from.unique()).union(set(data.point_to.unique()))
    
    ####VARIABLES####
    data['travel_on'] = m.binary_var_list([(row.point_from, row.point_to) for row in data.itertuples()], name = 'travel_on')
    
    ####OBJECTIVE####
    distance_traveled = m.sum([row.travel_on * row.distance for row in data.itertuples()]) 
    m.add_kpi(distance_traveled, "distance")
    
    m.minimize(distance_traveled)
    ####CONSTRAINTS####
    #force each location to only have one outgoing edge
    for start_point in all_locations:
        m.add_constraint(m.sum([row.travel_on for row in data[data.point_from == start_point].itertuples()]) == 1, \
            'outgoing_edge_%s' % start_point)

    #force each location to only have one incoming edge
    for end_point in all_locations:
        m.add_constraint(m.sum([row.travel_on for row in data[data.point_to == end_point].itertuples()]) == 1, \
            'incoming_edge_%s' % end_point)

    return m

#update the parameters, solve the model, and return the results
def update_and_solve(m, data, mipgap = 0.0001):
    m.parameters.mip.tolerances.mipgap.set(mipgap)
    m.parameters.timelimit = 600
    m.solve()
    data['travel_on_val'] = [var.solution_value for var in data.travel_on.values]

def add_cycle_breaking_constraints(m,subtour_id, data):
    g = networkx.DiGraph()
    g.add_edges_from([(row.point_from,row.point_to) for row in data[data.travel_on_val > 0.5].itertuples()])
    num_cycles = 0
    longest_cycle = None
    subtours = {}
    for cycle in networkx.simple_cycles(g):
        cycleLength = len(cycle)
        num_cycles += 1
        subtours[subtour_id] = cycle
        subtour_id += 1
    if num_cycles == 1: #finished
        return m.kpis_as_dict()['distance'], cycle, subtour_id
    else:
        for new_subtour in subtours.keys():
            cycle = subtours[new_subtour]
            idxs = data[(data.point_from.isin(cycle)) & (data.point_to.isin(cycle)) & (data.travel_on_val > 0.5)].index.values
            m.add_constraint(m.sum(data.loc[idxs,'travel_on'].values) <= len(cycle) - 1,\
                'subtour_elimintaion_%d' % subtour_id)
        return m.kpis_as_dict()['distance'], [], subtour_id
    
def distance_tsp(data, max_length):
    m = tsp(data, max_length)
    mipgap = 0.1
    subtour_id = 0
    iteracion = 0
    max_iter = 500
    estado_iter = ''
    update_and_solve(m, data, mipgap = mipgap)
    distance, tour, subtour_id = add_cycle_breaking_constraints(m,subtour_id, data)
    if len(tour) == max_length:
        mipgap = mipgap - 0.01
    while (len(tour) < max_length or mipgap > 0.009) and iteracion < max_iter:
        update_and_solve(m, data, mipgap = mipgap)
        distance, tour, subtour_id = add_cycle_breaking_constraints(m,subtour_id, data)
        if len(tour) == max_length:
            mipgap = 0
        iteracion = iteracion + 1
    if iteracion == max_iter:
        estado_iter = 'SIN SOLUCION'
    else:
        estado_iter='OPTIMO'
    return distance, tour, estado_iter

p = 20
df = pandas.DataFrame({'X':[random.randrange(100) for i in range(p)],'Y':[random.randrange(100) for i in range(p)]})

df.plot(kind='scatter', x='X', y='Y', figsize=(6, 6), color='darkblue')

perm = [x for x in itertools.permutations(range(p), 2)]

data = pandas.DataFrame(columns=['point_from','point_to','distance'])

for i in range(len(perm)):
    p1 = perm[i][0]
    p2 = perm[i][1]
    data.loc[i] = p1 , p2 , \
    round(numpy.lib.scimath.sqrt((df.loc[p1,'X'] - df.loc[p2,'X'])**2+(df.loc[p1,'Y'] - df.loc[p2,'Y'])**2),2)

max_length = len(set(data.point_from.unique()).union(set(data.point_to.unique())))
distance, tour, estado = distance_tsp(data, max_length)


def plot_tsp(df,points, style='bo-'):
    "Plot lines to connect a series of points."
    plt.plot([df.loc[p,'X'] for p in points], [df.loc[p,'Y'] for p in points], style)
    plt.axis('scaled'); plt.axis('off')
    plt.show()

first_point= tour[0]
tour.append(first_point)
plot_tsp(df,tour)