import argparse
import math
import networkx as nx
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    return parser.parse_args()


def parse_labels(file):
    with open(file, "r") as f:
        labels = f.readlines()
    solutions = []
    # iterate over chars
    for solution in labels:
        nodes = []
        for char in solution:
            if char == "Q":
                nodes.append(1)
            elif char == '.':
                nodes.append(0)
            else:
                pass
        solutions.append(nodes)
    return solutions


def n_queens_edges(N):
    edges = set()
    for r in range(N):
        for c in range(N):
            # for each call, add horizontal and vertical edges
            for i in range(N):
                if i != c:
                    edges.add((r * N + c, r * N + i))
                    edges.add((r * N + i, r * N + c))
                if i != r:
                    edges.add((r * N + c, i * N + c))
                    edges.add((i * N + c, r * N + c))
            # add diagonal edges
            for i in range(N):
                if 0 <= r + i < N and 0 <= c + i < N:
                    if i != 0:
                        edges.add((r * N + c, (r + i) * N + c + i))
                        edges.add(((r + i) * N + c + i, r * N + c))
                if 0 <= r + i < N and 0 <= c - i < N:
                    if i != 0:
                        edges.add((r * N + c, (r + i) * N + c - i))
                        edges.add(((r + i) * N + c - i, r * N + c))
                if 0 <= r - i < N and 0 <= c + i < N:
                    if i != 0:
                        edges.add((r * N + c, (r - i) * N + c + i))
                        edges.add(((r - i) * N + c + i, r * N + c))
                if 0 <= r - i < N and 0 <= c - i < N:
                    if i != 0:
                        edges.add((r * N + c, (r - i) * N + c - i))
                        edges.add(((r - i) * N + c - i, r * N + c))
    return edges

def draw_nqueens_graph(G):
    pos = nx.spectral_layout(G)
    # draw labels as red if == 1
    node_colors = ['red' if G.nodes[n]['label'] == 1 else 'skyblue' for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_colors)
    plt.show()


if __name__ == "__main__":
    '''
    Usage:
        python n_queens_to_nx.py --input_file 8_queens_train.txt --output_folder processed/train
    '''

    args = parse_args()
    input_file = args.input_file
    output_folder = args.output_folder

    solutions = parse_labels(input_file)
    N = int(math.sqrt(len(solutions[0])))

    for ii, node_labels in enumerate(solutions):
        edges = n_queens_edges(N)

        G = nx.Graph()
        G.add_nodes_from([i for i in range(N*N)])
        G.add_edges_from(edges)

        label_mapping = { v: node_labels[v] for v in G.nodes }
        nx.set_node_attributes(G, values=label_mapping, name='label')

        # draw nx graph
        if ii == 0:
            draw_nqueens_graph(G)

        # write graph to file
        output_file = f"{output_folder}/{ii}.gpickle" 
        nx.write_gpickle(G, output_file)
        print(f"Graph saved to {output_file}")

        # write node_labels to file
        output_file = f"{output_folder}/{ii}.result"
        with open(output_file, "w") as f:
            for label in node_labels:
                f.write(str(label) + "\n")
        print(f"Node labels saved to {output_file}")

        
    
