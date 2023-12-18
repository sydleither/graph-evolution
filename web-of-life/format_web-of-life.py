import csv
import os

import networkx as nx
import numpy as np


for file_name in os.listdir("original"):
    if file_name == "references.csv" or file_name == "README":
        continue
    
    matrix = np.loadtxt("original/{}".format(file_name), delimiter=',', dtype=str)
    rows, cols = matrix.shape

    G = nx.DiGraph()
    for i in range(1, rows):
        prey_name = matrix[i][0]
        for j in range(1, cols):
            pred_name = matrix[0][j]
            if pred_name == "\"Num. of hosts sampled\"":
                continue
            val = float(matrix[i][j])
            if val != 0:
                G.add_edge(pred_name, prey_name, weight=val)
    adjacency = nx.to_numpy_array(G).tolist()

    weights = [x for y in adjacency for x in y]
    max_weight = max(weights)
    if max_weight > 1:
        num_species = len(adjacency)
        adjacency = [[round(adjacency[i][j]/max_weight, 5) for j in range(num_species)] for i in range(num_species)]

    with open("formatted/{}".format(file_name), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(adjacency)