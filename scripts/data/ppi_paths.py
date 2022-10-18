import pandas as pd
import networkx as nx
import numpy as np
import pickle


prots = pd.read_csv("/Mounts/rbg-storage1/datasets/STRING/e_coli/511145.protein.info.v11.5.txt", sep = "\t")
prots["#string_protein_id"] = prots["#string_protein_id"].str.replace("511145\.", "", regex=True)
links = pd.read_csv("/Mounts/rbg-storage1/datasets/STRING/e_coli/511145.protein.links.v11.5.txt", sep = " ")
links["protein1"] = links["protein1"].str.replace("511145\.", "", regex=True)
links["protein2"] = links["protein2"].str.replace("511145\.", "", regex=True)


# Makes a graph of the PPI network and runs Floyd-Warshall algorithm (All pairs shortest path)
# Path weights are -log(weight) because we want the maximum multiplication along the paths
# idx2prot = {prot_id: idx for idx, prot_id in \
        #             zip(range(len(set(prots["#string_protein_id"].values))), \
        #                 set(prots["#string_protein_id"].values))}
edgelist = [i.tolist()[1:] for i in links.to_records()]
edgelist = [(p1, p2, {"weight": -np.log(w / 1000)}) for (p1, p2, w) in edgelist]
# edgelist = [(idx2prot[p1], idx2prot[p2], {"weight": w / 1000}) for (p1, p2, w) in edgelist]
G = nx.Graph(edgelist)
print("running Floyd Warshall")
predecessor, shortest_paths = nx.floyd_warshall_predecessor_and_distance(G)
print("Finished running Floyd Warshall")
clean_dict = {}
for k1 in shortest_paths:
    for k2 in shortest_paths[k1]:
        clean_dict.setdefault(k1, {})
        clean_dict[k1][k2] = np.exp(-1*shortest_paths[k1][k2])

pickle.dump(dict(clean_dict), open("path_lengths2.pkl", "wb"))

paths = {}
for k1 in shortest_paths:
    for k2 in shortest_paths[k1]:
        paths.setdefault(k1, {})
        paths[k1][k2] = nx.reconstruct_path(k1, k2, predecessor)

pickle.dump(dict(paths), open("predecessors.pkl", "wb"))
