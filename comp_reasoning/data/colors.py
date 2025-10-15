import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
import random
import os

class PairDataset(Dataset):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

    def __len__(self):
        return len(self.pairs)

    def mine_negative(self, pair):
        '''
        Return a negative pair of shape equal to one_hot_pair
        '''
        negatives = []
        i, j = pair
        oneh_i = F.one_hot(torch.tensor(i), num_classes=self.N).float()
        oneh_j = F.one_hot(torch.tensor(j), num_classes=self.N).float()

        # NEGATIVE CASE 1: same index on both one-hot vectors
        # choose between index i and j 
        rand_ind = random.choice([i, j])
        one_i = torch.ones_like(oneh_i) * 0
        one_j = torch.ones_like(oneh_j) * 0
        # random number between -0.99 and 1.0
        one_i[rand_ind] = random.uniform(0.5, 1.0)
        one_j[rand_ind] = random.uniform(0.5, 1.0)
        negatives.append(torch.cat((
            one_i, 
            one_j
        )))

        # NEGATIVE CASE 2: correct one hot vectors but with one zero entry as non zero
        neg = torch.cat([oneh_i.clone(), oneh_j.clone()], dim=0)
        neg_i = i
        neg_j = j + self.N  # offset for second half of the vector
        list_vals = [x for x in range(self.N * 2) if x != neg_i and x != neg_j]
        random_index = random.sample(list_vals, 1)[0]
        neg[random_index] = random.uniform(0.01, 1.0)
        negatives.append(neg)

        return negatives

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        i, j = pair
        oneh_i = F.one_hot(torch.tensor(i), num_classes=self.N).float()
        oneh_j = F.one_hot(torch.tensor(j), num_classes=self.N).float()
        one_hot_pair = torch.cat((oneh_i, oneh_j))

        negatives = self.mine_negative(pair)

        minus_mask = one_hot_pair == 0
        one_hot_pair[minus_mask] = 0.0
        for n_i in range(len(negatives)):
            minus_mask = negatives[n_i] == 0
            negatives[n_i][minus_mask] = 0.0

        return one_hot_pair, torch.stack(negatives, dim=0)
    

class AugmentedGraphDataset(Dataset):
    def __init__(self, graphs, max_colors, split_diameter=None, replace_colors=True):
        self.graphs = graphs
        self.max_colors = max_colors
        self.replace_colors = replace_colors
        if split_diameter is not None:
            new_graphs = []
            for graph in graphs:
                splitted_gs = partition_graph_diameter_N(graph, split_diameter)
                for g in splitted_gs:
                    if g.num_nodes > 1:
                        new_graphs.append(g)
            self.graphs = new_graphs
            

    def __len__(self):
        return len(self.graphs)

    def _mine_negative(self, data):
        '''
        neg = data.y.clone()
        neg = F.one_hot(neg, num_classes=self.max_colors).float()
        # select a random entry
        rand_index = random.randint(0, neg.shape[0] - 1)
        # random color that is not the current color
        list_vals = [x for x in range(self.max_colors) if x != data.y[rand_index].item()]
        random_color = random.sample(list_vals, 1)[0]
        # if p>0.5 add positive noise in the entry else add negative noise
        if random.uniform(0, 1) > 0.5:
            neg[rand_index, random_color] = random.uniform(0.01, 1.0)
        else:
            neg[rand_index, random_color] = random.uniform(-1.0, -0.01)
        return neg
        '''
        neg = data.y.clone()
        # select random node
        rand_index = random.randint(0, neg.shape[0] - 1)
        # get neighbors of the random node
        neighbors = data.edge_index[1][data.edge_index[0] == rand_index]
        # select a random neighbor
        if neighbors.numel() > 0:
            random_neighbor = neighbors[random.randint(0, neighbors.numel() - 1)]
            # replace the color of the random node with the color of the random neighbor
            neg[rand_index] = data.y[random_neighbor]
        else:
            # If no neighbors, just return the original colors
            pass
        neg = F.one_hot(neg, num_classes=self.max_colors).float()
        return neg

    def __getitem__(self, idx):
        data = self.graphs[idx]

        # Clone to avoid modifying the original graph
        data = data.clone()

        if hasattr(data, 'y') and self.replace_colors:
            data.y = replace_colors(data.y, self.max_colors)

        data.neg = self._mine_negative(data)
        data.y = F.one_hot(data.y, num_classes=self.max_colors).float()

        return data


def replace_colors(solution, max_colors):
    sol_colors = max(solution) + 1
    mapped_colors = random.sample(range(max_colors), sol_colors)
    for i in range(len(solution)):
        solution[i] = mapped_colors[solution[i]]
    return solution


def parse_graph_file(file_path):
    if file_path.endswith('.col'):
        num_nodes, edges = parse_col_file(file_path)
        return num_nodes, edges, None
    elif file_path.endswith('.graph'):
        num_nodes, edges, edge_weights, diff_edge, chrom_number = parse_tsplib_file(file_path)
        print(">", edges.shape)
        return num_nodes, edges, chrom_number
    else:
        raise ValueError("Unsupported file format. Please provide a .col or .graph file.")

def parse_col_file(file_path):
    edges = []
    max_node = 0
    set_edges = set()

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('e '):
                _, u, v = line.strip().split()
                u, v = int(u) - 1, int(v) - 1  # Convert to zero-based
                # Avoid duplicate edges
                if (u, v) in set_edges or (v, u) in set_edges:
                    continue
                edges.append((u, v))
                set_edges.add((u, v))
                max_node = max(max_node, u, v)

    edge_array = np.array(edges, dtype=int)
    num_nodes = max_node + 1  # Since index is zero-based
    return num_nodes, edge_array


def parse_tsplib_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = {}
    idx = 0

    data["EDGE_WEIGHT_MATRIX"] = -1 # Default value if not present

    # Parse header
    while ":" in lines[idx]:
        key, value = lines[idx].split(":")
        data[key.strip()] = value.strip()
        idx += 1

    # Parse EDGE_DATA_SECTION
    if lines[idx] == "EDGE_DATA_SECTION":
        idx += 1
        edge_data = []
        while lines[idx] != "-1":
            u, v = map(int, lines[idx].split())
            edge_data.append((u, v))
            idx += 1
        data["EDGE_DATA"] = edge_data
        idx += 1  # Skip -1

    # Parse EDGE_WEIGHT_SECTION
    if lines[idx] == "EDGE_WEIGHT_SECTION":
        idx += 1
        matrix = []
        dimension = int(data["DIMENSION"])
        for _ in range(dimension):
            row = list(map(float, lines[idx].split()))
            matrix.append(row)
            idx += 1
        data["EDGE_WEIGHT_MATRIX"] = matrix

    # Parse DIFF_EDGE
    if lines[idx] == "DIFF_EDGE":
        idx += 1
        data["DIFF_EDGE"] = tuple(map(int, lines[idx].split()))
        idx += 1

    # Parse CHROM_NUMBER
    if lines[idx] == "CHROM_NUMBER":
        idx += 1
        data["CHROM_NUMBER"] = int(lines[idx])
        idx += 1

    num_nodes = int(data["DIMENSION"])
    edges = np.array(data["EDGE_DATA"], dtype=int)
    edge_weights = np.array(data["EDGE_WEIGHT_MATRIX"], dtype=float)
    diff_edge = data.get("DIFF_EDGE", None)
    chrom_number = data.get("CHROM_NUMBER", None)
    return num_nodes, edges, edge_weights, diff_edge, chrom_number


def read_solution_file(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            node_id, solution = map(int, line.strip().split())
            result.append((node_id, solution))
    result.sort(key=lambda x: x[0])
    return [solution for _, solution in result]


def load_graphs_and_solutions(graphs_dir):
    result = []
    for filename in os.listdir(graphs_dir):
        if filename.endswith('.graph'):
            filepath = os.path.join(graphs_dir, filename)
            num_nodes, edges, edge_weights, diff_edge, chrom_number = parse_tsplib_file(filepath)
            solution = read_solution_file(filepath.replace('.graph', '.sol'))
            result.append((num_nodes, edges, solution))
    return result


def convert_to_pyg_data(graph_list):
    pyg_data_list = []

    for num_nodes, edge_index_np, node_labels in graph_list:
        edge_index = torch.tensor(edge_index_np, dtype=torch.long).t().contiguous()

        x = torch.ones((num_nodes, 1), dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        pyg_data_list.append(data)

    return pyg_data_list

    