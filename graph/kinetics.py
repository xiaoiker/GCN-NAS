import numpy as np
import sys

sys.path.extend(['../'])
from graph import tools
import networkx as nx

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

# Edge format: (origin, neighbor)
num_node = 18
self_link = [(i, i) for i in range(num_node)]
#inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
#         (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
#          (16, 14)]
inward = [(0, 1),(2, 1),(3, 2),(4, 3),(5, 1),(6, 5),(7, 6), (8, 2),(9, 8),(10, 9),(11, 5), (12, 11),(13, 12),  
               (14, 0), (15, 0), (16, 14), (17, 15)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


self_link13 = [(i, i) for i in range(13)]
inward_ori_index13 = [(0, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),(8, 2), (8, 5), (9, 8), (10, 9), (11, 8), (12, 11)]
inward13 = [(i - 1, j - 1) for (i, j) in inward_ori_index13]
outward13 = [(j, i) for (i, j) in inward13]


self_link9 = [(i, i) for i in range(9)]
inward_ori_index9 = [(2, 1), (3, 2), (4, 1), (5, 4), (6, 1), (7, 6),(8, 1), (9, 8)]
inward9 = [(i - 1, j - 1) for (i, j) in inward_ori_index9]
outward9 = [(j, i) for (i, j) in inward9]

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
        
    def get_A2515(self):# Actually , here convert 18 Nodes to 13 nodes. Just keep this function name.
        
        A_15 = np.zeros((18, 13))
        NTU_link_neighber = [[0,14,15,16,17],[1,0,5,2],[2,1,3],[3,2],[4],[5,1,6],[6,5],[7],[8,11],[9,10],[10],[11,12],[13]]
        #Ket_link_neighber = [[],[],[],[],]
        for i in range(13):
            #index = [a-1 for a in NTU_link_neighber[i]]
            index = NTU_link_neighber[i]
            A_15[index,i] = 1
    
        A_15 = tools.normalize_digraph(A_15)
        return A_15
        
    def get_A159(self):
        A_9 = np.zeros((13, 9))
        NTU_link_neighber = [[0,1,2,5,8],[5,6,7],[7],[2,3,4],[4],[8,11,12],[12],[8,9,10],[10]]
        #Ket_link_neighber = [[],[],[],[],]
        for i in range(9):
            #index = [a-1 for a in NTU_link_neighber[i]]
            index = NTU_link_neighber[i]
            A_9[index,i] = 1
    
        A_9 = tools.normalize_digraph(A_9)
        return A_9
        
    def get_A15(self):
        A = tools.get_spatial_graph(13, self_link13, inward13, outward13)
        return A 
                       
    def get_A9(self):
        A = tools.get_spatial_graph(9, self_link9, inward9, outward9)
        return A

if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    print('')
