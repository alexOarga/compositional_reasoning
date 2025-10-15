import torch
from torch.utils.data import Dataset
import pickle
import json


def one_hot_uppercase_string_torch(s):
    assert len(s) == 5 and s.isupper()
    idxs = [ord(c) - ord('A') for c in s]
    tt = torch.nn.functional.one_hot(torch.tensor(idxs), num_classes=26).float()
    tt = tt.reshape(-1)
    return tt

def idxs_to_uppercase_string(idxs):
    assert len(idxs) == 5
    chars = [chr(i + ord('A')) for i in idxs]
    return ''.join(chars)

def load_embeddings_dict(pickle_path):
    with open(pickle_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    emb_dict = {}
    for sentence, word, embedding in embeddings_data:
        emb_dict[sentence] = embedding
    return emb_dict

def _test_idx():
    test_idx = []
    for i in range(10):
        test_idx.append((i * 10) + 1)
        test_idx.append((i * 10) + 6)
    return test_idx

def load_instances(json_path, split='train'):
    data = json.load(open(json_path, 'r'))

    if split == 'test':
        test_idx = _test_idx()
        data = [item for i, item in enumerate(data) if i in test_idx]
    elif split == 'train':
        test_idx = _test_idx()
        data = [item for i, item in enumerate(data) if i not in test_idx]
    else:
        raise ValueError("split must be 'train' or 'test'")

    instances = []
    for item in data:
        desc = item[0]
        word = item[1]
        h_words = []
        h_desc = []
        for ii in range(5):
            w = word[ii*5:(ii+1)*5]
            w = ''.join(w)
            h_words.append(w)
            h_desc.append(desc[ii])
        v_words = []
        v_desc = []
        for ii in range(5):
            w = word[ii::5]
            w = ''.join(w)
            v_words.append(w)
            v_desc.append(desc[5 + ii])
        instances.append((h_desc, h_words, v_desc, v_words))
    return instances


def instance_to_torch(instance, emb_dict):
    h_desc, h_words, v_desc, v_words = instance
    h_word_tensors = [one_hot_uppercase_string_torch(w) for w in h_words]
    v_word_tensors = [one_hot_uppercase_string_torch(w) for w in v_words]
    
    h_embeddings = [emb_dict[desc] for desc in h_desc]
    v_embeddings = [emb_dict[desc] for desc in v_desc]
    
    h_embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in h_embeddings]
    v_embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in v_embeddings]
    
    return h_word_tensors, h_embeddings, v_word_tensors, v_embeddings


class CrosswordRowDataset(Dataset):
    def __init__(self, pickle_path):
        super().__init__()
        # Load the pickle file
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, word, embedding = self.data[idx]
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        word_tensor = one_hot_uppercase_string_torch(word)
        return word_tensor, embedding_tensor
    
    def vocabulary(self):
        return set(word for _, word, _ in self.data)