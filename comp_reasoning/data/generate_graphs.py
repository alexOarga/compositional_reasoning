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


def compute_graph_coloring(M, max_color=14):
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


def write_graph(Ma, Mw, diff_edge, filepath, int_weights=False, cn = 0):
    with open(filepath,"w") as out:

        n, m = Ma.shape[0], len(np.nonzero(Ma)[0])
        
        out.write('TYPE : TSP\n')

        out.write('DIMENSION: {n}\n'.format(n = n))

        out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
        out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')
        
        # List edges in the (generally not complete) graph
        out.write('EDGE_DATA_SECTION\n')
        for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
            out.write("{} {}\n".format(i,j))
        #end
        out.write('-1\n')

        # Write edge weights as a complete matrix
        out.write('EDGE_WEIGHT_SECTION\n')
        for i in range(n):
            if int_weights:
                out.write('\t'.join([ str(int(Mw[i,j])) for j in range(n)]))
            else:
                out.write('\t'.join([ str(float(Mw[i,j])) for j in range(n)]))
            #end
            out.write('\n')
        #end

        # Write diff edge
        out.write('DIFF_EDGE\n')
        out.write('{}\n'.format(' '.join(map(str,diff_edge))))
        if cn > 0:
          # Write chromatic number
          out.write('CHROM_NUMBER\n')
          out.write('{}\n'.format(cn))

        out.write('EOF\n')
    #end
#end


def generate_graph(graph_type, nmin, nmax, p_or_m, m=None, phase=None):
    if graph_type == "erdos_renyi":
        n = np.random.randint(nmin, nmax + 1)
        if phase is not None:
            p_or_m = np.log(n) / n
        G = nx.erdos_renyi_graph(n, p_or_m)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Compute average degree
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        average_degree = (2 * num_edges) / num_nodes
        print(f"Average Degree: {average_degree:.2f}")

        density = nx.density(G)
        print("Density:", density)

        adj_matrix = nx.to_numpy_array(G, dtype=int)
        return adj_matrix
    elif graph_type == "random_regular_expander_graph":
        n = np.random.randint(nmin, nmax + 1)
        G = nx.random_regular_expander_graph(n, p_or_m)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Compute average degree
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        average_degree = (2 * num_edges) / num_nodes
        print(f"Average Degree: {average_degree:.2f}")

        density = nx.density(G)
        print("Density:", density)

        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "chordal_cycle_graph":
        n = np.random.randint(nmin, nmax + 1)
        G = nx.chordal_cycle_graph(n)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "margulis_gabber_galil_graph":
        n = np.random.randint(nmin, nmax + 1)
        G = nx.margulis_gabber_galil_graph(n)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "random_regular_graph":
        n = np.random.randint(nmin, nmax + 1)
        G = nx.random_regular_graph(p_or_m, n)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "powerlaw_cluster_graph":
        n = np.random.randint(nmin, nmax + 1)
        G = nx.powerlaw_cluster_graph(n, m, p_or_m)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "watts_strogatz_graph":
        n = np.random.randint(nmin, nmax + 1)
        G = nx.watts_strogatz_graph(n, m, p_or_m)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "barabasi_albert_graph":
        n = np.random.randint(nmin, nmax + 1)
        G = nx.barabasi_albert_graph(n, int(p_or_m))
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "paley_graph":
        n = nmin
        G = nx.paley_graph(n)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "random_geometric_graph":
        n = np.random.randint(nmin, nmax + 1)
        G = nx.random_geometric_graph(n, p_or_m)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "stochastic_block_model":
        n = np.random.randint(nmin, nmax + 1)
        sizes = [n // 2, n // 2]
        p = [[p_or_m, p_or_m], [p_or_m, p_or_m]]
        G = nx.stochastic_block_model(sizes, p)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    elif graph_type == "complete_graph":
        G = nx.complete_graph(nmin)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj_matrix = nx.to_numpy_array(G, dtype=int)
    else:
        raise ValueError("Unsupported graph type")
    return adj_matrix

def main():
    parser = argparse.ArgumentParser(description="Generate a random graph and print its adjacency matrix.")
    parser.add_argument("graph_type", help="Type of graph to generate.")
    parser.add_argument("nsamples", type=int, help="Number of samples")
    parser.add_argument("nmin", type=int, help="Minimum number of nodes.")
    parser.add_argument("nmax", type=int, help="Maximum number of nodes.")
    parser.add_argument("p", type=float, help="Edge probability or density")
    parser.add_argument("--m", type=int, help="Parameter for some graph generators.")
    # flag args
    parser.add_argument("--phase", action="store_true", help="Use phase for Erdos-Renyi graph generation.")

    args = parser.parse_args()

    if args.graph_type == 'paley_graph':
        # list primes between nmin and nmax
        primes = []
        for i in range(args.nmin, args.nmax + 1):
            if all(i % j != 0 for j in range(2, int(i**0.5) + 1)):
                primes.append(i)

    nmin = args.nmin
    nmax = args.nmax

    for i in range(args.nsamples):
        # Generate a random graph
        if args.graph_type == 'paley_graph':
            args.nmin = primes[i]
            args.nmax = primes[i]
        if args.graph_type == 'complete_graph':
            args.nmin = [j for j in range(nmin, nmax + 1)][i]
            args.nmax = args.nmin
        A = generate_graph(args.graph_type, args.nmin, args.nmax, args.p, args.m, args.phase)

        color = compute_graph_coloring(A)
        print(f"Chromatic Number: {color}")

        # Write graph to file
        filepath = f"graphs3/{args.graph_type}_{args.nmin}_{args.nmax}_{args.p}_{i}.graph"
        write_graph(A, A, (0, 0), filepath, int_weights=False, cn = color)


if __name__ == "__main__":
    main()