import cnfgen
import random


NUM_INSTANCES = 5000
OUTPUT_DIR = './3sat/train'

K = 3 # 3-SAT problem
N_MIN = 10
N_MAX = 20
#M = ... # M is choosen in the transition phase

count = 0
while count < NUM_INSTANCES:
    n = random.randint(N_MIN, N_MAX)
    m = int(4.258 * n)
    F = cnfgen.RandomKCNF(K, n, m)
    satisfiable, assignment = F.solve(cmd='minisat -no-pre')
    if satisfiable:
        F.to_file(f'{OUTPUT_DIR}/instance_{count}.cnf')
        with open(f'{OUTPUT_DIR}/instance_{count}.sol', 'w') as f:
            f.write(' '.join(map(str, assignment)))
        count += 1
    