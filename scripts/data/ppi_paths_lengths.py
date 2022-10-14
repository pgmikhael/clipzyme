import pandas as pd
import networkx as nx
import numpy as np
import pickle


prots = pd.read_csv("/Mounts/rbg-storage1/datasets/STRING/e_coli/511145.protein.info.v11.5.txt", sep = "\t")
prots["#string_protein_id"] = prots["#string_protein_id"].str.replace("511145\.", "", regex=True)
links = pd.read_csv("/Mounts/rbg-storage1/datasets/STRING/e_coli/511145.protein.links.v11.5.txt", sep = " ")
links["protein1"] = links["protein1"].str.replace("511145\.", "", regex=True)
links["protein2"] = links["protein2"].str.replace("511145\.", "", regex=True)

import argparse
parser = argparse.ArgumentParser(description='Finds shortest paths on G')
parser.add_argument("--threshold", help="A cutoff for edge probabilities in STRINGdb", type=float, default=0.5)
args = parser.parse_args()

edgelist = [i.tolist()[1:] for i in links.to_records()]
edgelist = [(p1, p2, {"weight": w}) for (p1, p2, w) in edgelist if w >= args.threshold]

G = nx.Graph(edgelist)
print("running Floyd Warshall")
shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
print("Finished running Floyd Warshall")
clean_dict = {}
for k1 in shortest_paths:
    for k2 in shortest_paths[k1]:
        clean_dict.setdefault(k1, {})
        clean_dict[k1][k2] = shortest_paths[k1][k2]

pickle.dump(dict(clean_dict), open("shortest_paths_hops.pkl", "wb"))
