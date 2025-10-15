import os
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


def parse_tsplib_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = {}
    idx = 0

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

    # return num nodes and array of edges
    num_nodes = int(data["DIMENSION"])
    edges = np.array(data["EDGE_DATA"], dtype=int)
    edge_weights = np.array(data["EDGE_WEIGHT_MATRIX"], dtype=float)
    diff_edge = data.get("DIFF_EDGE", None)
    chrom_number = data.get("CHROM_NUMBER", None)
    return num_nodes, edges, edge_weights, diff_edge, chrom_number


def write_solution_to_file(solution, solution_filename):
    with open(solution_filename, "w") as f:
        for node, color in solution.items():
            f.write(f"{node} {color}\n")
    print(f"Solution written to {solution_filename}")   

def main():
    parser = argparse.ArgumentParser(description="Generate solutions for graph coloring.")
    parser.add_argument("dir", help="Directory containing graph files.")

    args = parser.parse_args()

    # iterate over ".graph" files in the directory
    for filename in os.listdir(args.dir):
        if filename.endswith(".graph"):
            
            solution_filename = os.path.join(args.dir, filename.replace(".graph", ".sol"))
            if os.path.exists(solution_filename):
                print(f"Skipping {filename} as solution already exists.")
                continue
            else:
                print(f"Processing {filename}...")

            filepath = os.path.join(args.dir, filename)
            num_nodes, edges, edge_weights, diff_edge, chrom_number = parse_tsplib_file(filepath)
            solution = solve_csp(edge_weights, chrom_number)
            write_solution_to_file(solution, solution_filename)


if __name__ == "__main__":
    main()
