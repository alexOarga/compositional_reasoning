import random
import torch
import cnfgen
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from itertools import product


class DimacsSATOneSingleClauseDataset(Dataset):
    def __init__(self, directory, is_train):
        self.directory = directory
        self.is_train = is_train
        self.clauses = []
        self.solutions = []
        self.data = []
        self.load_data(directory)
        self.process_clauses()

    def load_data(self, directory):
        self.max_variables = 0

        # list all '.cnf' files in the directory
        files = [f for f in os.listdir(directory) if f.endswith('.cnf')]

        solutions = []
        if self.is_train:
            solutions = [f.replace('.cnf', '.sol') for f in files]

        # load .cnf files
        for i, file in enumerate(files):
            cnf = cnfgen.utils.parsedimacs.from_dimacs_file(cnfgen.formula.cnf.CNF, fileorname=os.path.join(directory, file))
            if self.is_train:
                with open(os.path.join(directory, solutions[i]), 'r') as s:
                    solution = list(map(int, s.read().split()))
                self.solutions.append(solution)
            self.clauses.append(cnf.clauses())
            self.max_variables = max(self.max_variables, cnf.number_of_variables())

        self.max_clauses = max([len(clause) for clause in self.clauses])

    def process_clauses(self):
        for i in range(len(self.clauses)):
            for j in range(len(self.clauses[i])):
                clause = self.clauses[i][j] # e.g. -1, 2,-7
                clause = torch.tensor(clause)
                solution = self.solutions[i]
                # get solution (0=False, 1=True) values for clause variables
                idx = torch.abs(clause.clone().long()) - 1
                #clause_solution = solution[idx]
                clause_solution = torch.gather(torch.tensor(solution), 0, idx)
                clause_solution[clause_solution > 0] = 1
                clause_solution[clause_solution < 0] = 0
                # get negative mask (0=Not negated, 1=Negated) for clause variables
                negation = (clause < 0).int()
                # use as negative sample the opposite clause sign (negative -> 1, positive -> 0)
                negative_sample = torch.sign(clause)
                negative_sample[negative_sample > 0] = 0
                negative_sample[negative_sample < 0] = 1
                self.data.append((clause_solution, negation, negative_sample))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class DimacsSATDataset(Dataset):
    def __init__(self, directory, is_train):
        self.directory = directory
        self.is_train = is_train
        self.clauses = []
        self.solutions = []
        self.load_data(directory)

    def load_data(self, directory):
        files = [f for f in os.listdir(directory) if f.endswith('.cnf')]
        solutions = []
        if self.is_train:
            solutions = [f.replace('.cnf', '.sol') for f in files]

        self.max_variables = 0

        for i, file in enumerate(files):
            cnf = cnfgen.utils.parsedimacs.from_dimacs_file(cnfgen.formula.cnf.CNF, fileorname=os.path.join(directory, file))
            if self.is_train:
                with open(os.path.join(directory, solutions[i]), 'r') as s:
                    solution = list(map(int, s.read().split()))
                self.solutions.append(solution)
            self.clauses.append(cnf.clauses())
            self.max_variables = max(self.max_variables, cnf.number_of_variables())

        self.max_clauses = max([len(clause) for clause in self.clauses])

    def __len__(self):
        return len(self.clauses)
    
    def __getitem__(self, idx):
        c = [torch.tensor(cc) for cc in self.clauses[idx]]
        mask_clause = [True] * len(self.clauses[idx]) + [False] * (self.max_clauses - len(self.clauses[idx]))
        mask_clause = torch.tensor(mask_clause)
        s = None
        if self.is_train:
            s = torch.tensor(self.solutions[idx]) 
        len_var = torch.max(torch.abs(torch.cat(c, dim=0)))
        mask_var = [True] * len_var + [False] * (self.max_variables - len_var)
        mask_var = torch.tensor(mask_var)
        neg = None
        if self.is_train:
            neg = s.clone()
            # indexes where mask_var is True
            idx = torch.where(mask_clause)[0]
            # sample a random index
            i = random.choice(idx)
            clause_neg = c[i]
            sign_clause_neg = torch.sign(clause_neg)
            # we negate the sign of the clause: positive -> -1, negative -> 1
            sign_clause_neg = -sign_clause_neg
            # NOTE: the sign of the clause indicates a non-satisfied assignment -> therefore, it is considered as a negative sample
            # we assing the value to the variables
            variables_neg = torch.abs(clause_neg) - 1
            neg[variables_neg] = torch.abs(neg[variables_neg]) * sign_clause_neg
        return c, mask_clause, s, mask_var, neg

    def collate_fn(self, batch):
        '''
        collate clauses (list of tensor) into a list of tensor (with tensor stacked in first dimension)
        '''
        stacked_clauses = [[] for _ in range(self.max_clauses)]
        for b in batch:
            c, mask_clause, s, mask_var, neg = b
            for i in range(len(c)):
                stacked_clauses[i].append(c[i])
            # pad the rest of the clauses with zero clauses
            for i in range(len(c), self.max_clauses):
                stacked_clauses[i].append(torch.zeros_like(c[0]))
        for i in range(len(stacked_clauses)):
            stacked_clauses[i] = torch.stack(stacked_clauses[i], dim=0)
        mask_clause = torch.stack([b[1] for b in batch], dim=0)
        mask_var = torch.stack([b[3] for b in batch], dim=0)

        sol = None
        neg = None
        if self.is_train:
            for i, b in enumerate(batch):
                batch[i] = list(b)
                s = b[2]
                s = torch.cat([s, torch.zeros(self.max_variables - s.shape[0], dtype=s.dtype)])
                batch[i][2] = s
            sol = torch.stack([b[2] for b in batch], dim=0)
            for i, b in enumerate(batch):
                neg = b[4]
                neg = torch.cat([neg, torch.zeros(self.max_variables - neg.shape[0], dtype=neg.dtype)])
                batch[i][4] = neg
            neg = torch.stack([b[4] for b in batch], dim=0)

        return stacked_clauses, mask_clause, sol, mask_var, neg