import random
import numpy as np
from copy import deepcopy 
def powerset(n):
    powerset = []
    for i in range(1 << n):
        powerset.append(tuple([int(_) for _ in np.binary_repr(i, width=n)]))
    return powerset

def aj_to_adj_list(adj_mat):
    n_causes = len(adj_mat)
    adj_list = dict()
    for (node, succesors) in enumerate(adj_mat):
        adj_list[node] = [(i + n_causes, succesors[i]) for i in range(n_causes) if succesors[i] > 0]
    return adj_list

def remove_edges(adj_list, prob=0.85):
    for node in adj_list:
        r = np.random.uniform()
        if r > prob:
            np.random.shuffle(adj_list[node])
            if len(adj_list[node]) > 0:
                adj_list[node].pop()
    return adj_list

def to_wrong_graph(adj_list, n_effects, prob=0.6):
    for node in adj_list:
        r = np.random.uniform()
        if prob / 2 < r < prob:
            np.random.shuffle(adj_list[node])
            if len(adj_list[node]) > 0:
                adj_list[node].pop()
        elif r < prob / 2:
            effects = np.arange(n_effects, 2 * n_effects)
            np.random.shuffle(effects)
            for effect in effects:
                in_effects = False
                for suc in adj_list[node]:
                    if effect == suc[0]: in_effects = True
                if not in_effects: 
                    adj_list[node].append((effect, 1))
                    break
    return adj_list

def del_edges(adj_list, elimination_percentage=0.0):
    n_edges = 0
    for node in adj_list:
        n_edges += len(adj_list[node])
        np.random.shuffle(adj_list[node])
    edges_to_rmv = int(n_edges * elimination_percentage)
    shuffled_nodes = list(adj_list.keys())
    np.random.shuffle(shuffled_nodes)
    i = 0
    while i < edges_to_rmv:
        for node in shuffled_nodes:
            if len(adj_list[node]) > 0:
                adj_list[node].pop()
                i += 1
            if i == edges_to_rmv: break
    return adj_list

def shuffle_aj_mat(adj_list, percentage=0.0):
    n_edges = 0
    for node in adj_list:
        n_edges += len(adj_list[node])
        np.random.shuffle(adj_list[node])
    edges_to_rmv = int(n_edges * percentage)
    shuffled_nodes = list(adj_list.keys())
    np.random.shuffle(shuffled_nodes)
    i = 0
    rmv_edges_parents = dict()
    while i < edges_to_rmv:
        for node in shuffled_nodes:
            if len(adj_list[node]) > 0:
                rmv_edges_parents[adj_list[node].pop()[0]] = node
                i += 1
            if i == edges_to_rmv: break
    shuffled_edges = list(rmv_edges_parents.keys())
    np.random.shuffle(shuffled_edges)
    np.random.shuffle(shuffled_nodes)
    shuffled_edges = shuffled_edges[:len(shuffled_edges) // 2]
    i = 0
    while len(shuffled_edges) > 0:
        edge = shuffled_edges.pop()
        if rmv_edges_parents[edge] != shuffled_nodes[i]:
            adj_list[shuffled_nodes[i]].append((edge, 1.0))
        else:
            shuffled_edges.append(edge)
        i += 1
        i %= len(shuffled_nodes)
    return adj_list

if __name__ == "__main__":
    a  = np.array([np.zeros(7) for _ in range(7)])
    for i in range(len(a)):
        for j in range(len(a[i])):
            if np.random.uniform() > 0.75:
                a[i, j] = 1
    print(a)
    adj_list = aj_to_adj_list(a)
    print("ADJ")
    print(adj_list)
    print("ADJ INCOMPLETA")
    print(del_edges(deepcopy(adj_list), 0.5))
    print("ADJ WRONG")
    print(shuffle_aj_mat(deepcopy(adj_list), 1.0))
    
    # print("INCOMPLETA")
    # adj_list_incompleta = remove_edges(deepcopy(adj_list))
    # print(adj_list_incompleta)
    # print("INCORRECTA")
    # adj_list_incorrecta = to_wrong_graph(deepcopy(adj_list), len(a))
    # print(adj_list_incorrecta)
    # print("ADJ")
    # print(adj_list)