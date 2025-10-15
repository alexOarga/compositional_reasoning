import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import to_networkx
import networkx as nx


def partition_graph_diameter_N(data: Data, N: int):
    assert N >= 1, "Diameter must be at least 1"
    
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    partitions = []

    # Convert to NetworkX for easy shortest path checks
    G_nx = to_networkx(data, to_undirected=True)
    
    while not visited.all():
        # Find an unvisited seed node
        seed = (~visited).nonzero(as_tuple=False)[0].item()
        
        # Find nodes within radius = floor(N/2)
        radius = N // 2
        
        # BFS subgraph
        sub_nodes, sub_edge_idx, mapping, edge_mask = k_hop_subgraph(seed, radius, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        sub_data = Data(x=data.x[sub_nodes] if data.x is not None else None,
                        edge_index=sub_edge_idx,
                        edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
                        orig_node_ids=sub_nodes,
                        y= data.y[sub_nodes] if data.y is not None else None,
                        t= data.t[sub_nodes] if hasattr(data, 't') and data.t is not None else None,
                        e= data.e[sub_nodes] if hasattr(data, 'e') and data.e is not None else None
                        )
        partitions.append(sub_data)
        
        # Update visited with internal nodes (not boundary)
        for node in sub_nodes:
            neighbors = list(G_nx.neighbors(node.item()))
            if all(n in sub_nodes.tolist() for n in neighbors):
                visited[node] = True

    return partitions