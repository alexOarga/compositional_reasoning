import torch
from torch.utils.data import Dataset
    

class NQueensRowsDataset(Dataset):
    def __init__(self, file_path, split='train'):
        """
        Args:
            file_path (str): Path to the N-Queens solutions file.
        """
        self.file_path = file_path
        
        with open(file_path, "r") as f:
            self.solutions = [line.strip() for line in f if line.strip()]
        
        if split == 'train':
            self.solutions = self.solutions[:len(self.solutions)//2]
        elif split == 'valid':
            num_val = len(self.solutions)//4
            self.solutions = self.solutions[len(self.solutions)//2 : (len(self.solutions)//2)+num_val]
        elif split == 'test':
            num_val = len(self.solutions)//4
            self.solutions = self.solutions[(len(self.solutions)//2)+num_val:]
        elif split == 'all':
            pass
        else:
            raise ValueError("Invalid split value. Must be one of 'train', 'valid', 'test', or 'all'.")

        # Infer board size N from the first solution
        self.N = int(len(self.solutions[0]) ** 0.5)

        for i in range(len(self.solutions)):
            self.solutions[i] = torch.tensor([1 if c == 'Q' else 0 for c in self.solutions[i]], dtype=torch.float32)
            self.solutions[i] = self.solutions[i].view(self.N, self.N)

        self.data = []
        for i in range(len(self.solutions)):
            for r in range(self.N):
                self.data.append(self.solutions[i][r])
            for c in range(self.N):
                self.data.append(self.solutions[i][:, c])

    def negative_sample(self, sample):
        # Random increase where value is 0
        x_neg = sample.clone() # random increase
        random_idx = torch.where(x_neg == 0)[0]
        random_idx = random_idx[torch.randperm(random_idx.size(0))][:1]
        random_perturbation = torch.randn_like(x_neg) * 0.5 + 0.5
        x_neg[random_idx] += random_perturbation[random_idx]

        # Random decrease
        x_neg_2 = sample.clone() # random decrease
        idx = torch.round(x_neg_2) == 1            
        random_perturbation = torch.randn_like(x_neg_2) * 0.5 + 0.5
        x_neg_2[idx] -= random_perturbation[idx]

        x_neg_3 = torch.zeros_like(sample) # No queen placed

        x_0_neg = torch.stack([x_neg, x_neg_2, x_neg_3], dim=0)
        if x_0_neg.ndim == 1:
            x_0_neg = x_0_neg.unsqueeze(0) # otherwise you shuffle the elements of the tensor
        # subsample the negatives
        x_0_neg = x_0_neg[torch.randperm(x_0_neg.shape[0])][:1] # only one negative sample per positive
        # reduce the dimension back to 1
        x_0_neg = x_0_neg.squeeze(0)
        return x_0_neg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d, n = self.data[idx], self.negative_sample(self.data[idx])
        return d, n
    

class NQueensDiagonalsDataset(Dataset):
    def __init__(self, file_path, pad=0, split='train'):
        """
        Args:
            file_path (str): Path to the N-Queens solutions file.
            pad (int): Padding to add around the board.
        """
        self.file_path = file_path
        self.pad = pad
        
        with open(file_path, "r") as f:
            self.solutions = [line.strip() for line in f if line.strip()]
        
        if split == 'train':
            self.solutions = self.solutions[:len(self.solutions)//2]
        elif split == 'valid':
            num_val = len(self.solutions)//4
            self.solutions = self.solutions[len(self.solutions)//2 : (len(self.solutions)//2)+num_val]
        elif split == 'test':
            num_val = len(self.solutions)//4
            self.solutions = self.solutions[(len(self.solutions)//2)+num_val:]
        elif split == 'all':
            pass
        else:
            raise ValueError("Invalid split value. Must be one of 'train', 'valid', 'test', or 'all'.")

        # Infer board size N from the first solution
        self.N = int(len(self.solutions[0]) ** 0.5)
        
        for i in range(len(self.solutions)):
            self.solutions[i] = torch.tensor([1 if c == 'Q' else 0 for c in self.solutions[i]], dtype=torch.float32)
            self.solutions[i] = self.solutions[i].view(self.N, self.N)
        
        # Generate diagonal tensors
        self.diagonals = []
        for i in range(len(self.solutions)):
            for d in range(-self.N + 1, self.N):
                diag = []
                for r in range(self.N):
                    c = r + d
                    if 0 <= c < self.N:
                        diag.append(self.solutions[i][r, c].item())
                while len(diag) < self.N:
                    diag.append(pad)
                self.diagonals.append(torch.tensor(diag, dtype=torch.float32))

    def negative_sample(self, sample):
        # NOTE: only random increase is applied because rows can be non-zero
        # Random increase where the value is 0
        x_neg = sample.clone() # random increase
        random_idx = torch.where(x_neg == 0)[0]
        random_idx = random_idx[torch.randperm(random_idx.size(0))][:1]
        random_perturbation = torch.randn_like(x_neg) * 0.5 + 0.5
        x_neg[random_idx] += random_perturbation[random_idx]

        x_0_neg = torch.stack([x_neg], dim=0)
        if x_0_neg.ndim == 1:
            x_0_neg = x_0_neg.unsqueeze(0) # otherwise you shuffle the elements of the tensor
        # subsample negatives
        x_0_neg = x_0_neg[torch.randperm(x_0_neg.shape[0])][:1] # only one negative sample per positive
        # reduce the dimension back to 1
        x_0_neg = x_0_neg.squeeze(0)
        return x_0_neg

    def __len__(self):
        return len(self.diagonals)
    
    def __getitem__(self, idx):
        return self.diagonals[idx], self.negative_sample(self.diagonals[idx])
    