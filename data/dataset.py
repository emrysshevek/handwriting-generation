import string
import json

import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

from utils.string_utils import str2label
from utils.character_set import make_char_set

start_token = '<start>'
end_token = '<end>'
unknown_token = '<unk>'

train_path = 'data/training_lines.json'


def collate(samples):
    labels = pad_sequence([label for label, _ in samples], batch_first=True)
    lengths = torch.tensor([length for _, length in samples])

    return labels, lengths


class BaseDataset(data.Dataset):

    def __init__(self):
        # self.char_to_idx = None
        # self.idx_to_char = None
        self.char_to_idx, self.idx_to_char, _ = make_char_set([train_path])
        self.n_chars = len(self.char_to_idx)
        self.data = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        label = str2label(self.data[index], self.char_to_idx)
        # label = str2label([start_token] + [x for x in self.data[index]] + [end_token], self.char_to_idx, self.char_to_idx[unknown_token])
        label_length = len(label)
        return torch.tensor(label, dtype=torch.long), label_length


class WordsDataset(BaseDataset):

    def __init__(self, path):
        super().__init__()

        with open(path, 'r') as fp:
            self.data = fp.read().split()

        # self.chars = set("".join(self.data))
        #
        # self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}
        # self.char_to_idx[start_token] = len(self.char_to_idx) + 1
        # self.char_to_idx[end_token] = len(self.char_to_idx) + 1
        # self.char_to_idx[unknown_token] = len(self.char_to_idx) + 1
        # self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        #
        # self.n_chars = len(self.char_to_idx)

    def __len__(self):
        return len(self.data)


class LetterDataset(BaseDataset):

    def __init__(self, size=26):
        super().__init__()

        self.size = size
        self.data = [''.join(['a']*i) for i in range(size)]

        # self.chars = string.ascii_lowercase
        #
        # self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}
        # self.char_to_idx[start_token] = len(self.char_to_idx) + 1
        # self.char_to_idx[end_token] = len(self.char_to_idx) + 1
        # self.char_to_idx[unknown_token] = len(self.char_to_idx) + 1
        # self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        #
        # self.n_chars = len(self.char_to_idx)

    def __len__(self):
        return self.size
