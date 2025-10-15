from pyeda.inter import *
import random

def solve_n_queens(N, train_split=0.5, valid_split=0.1, test_split=0.4):
    X = exprvars('x', N, N)
    
    # Row and Column Constraints
    R = And(*[OneHot(*[X[r, c] for c in range(N)]) for r in range(N)])
    C = And(*[OneHot(*[X[r, c] for r in range(N)]) for c in range(N)])

    # Diagonal Constraints
    starts_lr = [(i, 0) for i in range(N-1, 0, -1)] + [(0, i) for i in range(N)]
    starts_rl = [(i, N-1) for i in range(N-1, -1, -1)] + [(0, i) for i in range(N-1, 0, -1)]

    DLR = And(*[OneHot0(*[X[r, c] for r, c in diag])
                for diag in [[(r + i, c + i) for i in range(N - max(r, c))]
                             for r, c in starts_lr]])
    
    DRL = And(*[OneHot0(*[X[r, c] for r, c in diag])
                for diag in [[(r + i, c - i) for i in range(min(N - r, c + 1))]
                             for r, c in starts_rl]])

    # Combine all constraints
    S = R & C & DLR & DRL

    # Generate solutions
    solutions = []
    for soln in S.satisfy_all():
        board = ''.join(['Q' if soln[X[r, c]] else '.' for r in range(N) for c in range(N)])
        solutions.append(board)

    len_train = 1
    len_valid = 0
    len_test = len(solutions) - len_train - len_valid

    random.seed(0)
    random.shuffle(solutions)

    train_solutions = solutions[:len_train]
    valid_solutions = solutions[len_train:len_train+len_valid]
    test_solutions = solutions[-len_test:]

    for split in ["train", "valid", "test"]:
        filename = f"{N}_queens_{split}.txt"
        with open(filename, "w") as f:
            if split == "train":
                for solution in train_solutions:
                    f.write(solution + "\n")
            elif split == "valid":
                for solution in valid_solutions:
                    f.write(solution + "\n")
            else:
                for solution in test_solutions:
                    f.write(solution + "\n")
        print(f"Solutions saved to {filename}")

if __name__ == "__main__":
    N = int(input("Enter board size N: "))
    solve_n_queens(N)