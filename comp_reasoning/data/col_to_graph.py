import argparse
import networkx as nx
import numpy as np
from ortools.sat.python import cp_model


def solve_csp(M, n_colors, nmin=10):
  model = cp_model.CpModel()
  N = len(M)
  variables = []
  
  variables = [ model.NewIntVar(0, n_colors-1, '{i}'.format(i=i)) for i in range(N) ]
  
  for i in range(N):
    for j in range(i+1,N):
      if M[i][j] == 1:
        model.Add( variables[i] != variables [j] )
        
  solver = cp_model.CpSolver()
  solver.parameters.max_time_in_seconds = int( ((10.0 / nmin) * N) )
  status = solver.Solve(model)
  
  if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL :
    solution = dict()
    for k in range(N):
      solution[k] = solver.Value(variables[k])
    return solution
  elif status == cp_model.INFEASIBLE:
    return None
  else:
    raise Exception("CSP is unsure about the problem")


def compute_graph_coloring(M, max_color=15):
    # Find the chromatic number of the graph
    for n_colors in range(2, max_color + 1):
        try:
            solution = solve_csp(M, n_colors)
            if solution is not None:
                # If a solution is found, return the number of colors used
                return n_colors
        except Exception as e:
            pass
    raise Exception("CSP is unsure about the problem")


def write_graph(Ma, Mw, diff_edge, filepath, int_weights=False, cn=0, skip_edge_weights=False):
    with open(filepath, "w") as out:
        n = Ma.shape[0]

        out.write('TYPE : TSP\n')
        out.write('DIMENSION: {n}\n'.format(n=n))
        out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
        out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX\n')

        # List edges
        out.write('EDGE_DATA_SECTION\n')
        for (i, j) in zip(*np.nonzero(Ma)):
            out.write(f"{i} {j}\n")
        out.write('-1\n')

        # Write edge weights unless skipped
        if not skip_edge_weights:
            out.write('EDGE_WEIGHT_SECTION\n')
            for i in range(n):
                if int_weights:
                    out.write('\t'.join([str(int(Mw[i, j])) for j in range(n)]))
                else:
                    out.write('\t'.join([str(float(Mw[i, j])) for j in range(n)]))
                out.write('\n')

        # Write diff edge
        out.write('DIFF_EDGE\n')
        out.write('{}\n'.format(' '.join(map(str, diff_edge))))

        # Write chromatic number if specified
        if cn >= 0:
            out.write('CHROM_NUMBER\n')
            out.write('{}\n'.format(cn))

        out.write('EOF\n')

def parse_col_file(file_path):
    edges = []
    max_node = 0
    set_edges = set()

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('e '):
                _, u, v = line.strip().split()
                u, v = int(u) - 1, int(v) - 1  # Convert to zero-based
                if (u, v) in set_edges or (v, u) in set_edges:
                    continue
                edges.append((u, v))
                set_edges.add((u, v))
                max_node = max(max_node, u, v)

    edge_array = np.array(edges, dtype=int)
    num_nodes = max_node + 1
    return num_nodes, edge_array

def compute_graph_coloring(A):
    G = nx.from_numpy_array(A)
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    num_colors = max(coloring.values()) + 1
    return num_colors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a COLOR file to a graph format.')
    parser.add_argument('input_dir', type=str, help='Input folder with .col files')
    parser.add_argument('output_dir', type=str, help='Output folder to store .graph files')
    parser.add_argument('--skip-coloring', action='store_true', help='Skip computing graph coloring')
    parser.add_argument('--skip-edge-weights', action='store_true', help='Skip writing EDGE_WEIGHT_SECTION')
    args = parser.parse_args()

    import os
    import glob

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for col_file in glob.glob(os.path.join(args.input_dir, '*.col')):
        filepath = os.path.join(args.output_dir, os.path.basename(col_file).replace('.col', '.graph'))
        if os.path.exists(filepath):
            print("Graph file already exists, skipping:", filepath)
            continue

        print("Processing file:", col_file)
        num_nodes, edge_array = parse_col_file(col_file)

        G = nx.Graph()
        G.add_edges_from(edge_array)
        A = nx.to_numpy_array(G, dtype=int)

        if args.skip_coloring:
            color = 0
        else:
            color = compute_graph_coloring(A)

        write_graph(A, A, (0, 0), filepath, int_weights=False, cn=color, skip_edge_weights=args.skip_edge_weights)
