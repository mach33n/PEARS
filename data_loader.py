import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torchtext.transforms as transforms
from torch.hub import load_state_dict_from_url

import pandas as pd
import numpy as np

from math import floor, ceil

class NLPDataset(Dataset):
    # *add should be a list of any additional labels to be used from the dataset
    def __init__(self, file_name, train, label, cols, *add):
        nlp_df = pd.read_csv(file_name, usecols=cols)
        nlp_df = nlp_df.dropna(subset=[train, label])
        nlp_df = nlp_df.reset_index(drop=True)
        self.x_train = nlp_df[train]
        self.y_train = nlp_df[label]
        for label in add:
            self.label = nlp_df[label]

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx], self.y_train[idx]


class AmazonDataLoader:
    def __init__(self, batch_size=59, max_seq=256, shuffle=True):
        ## Custom Parameters
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.bos_idx = 0
        self.pad_idx = 1
        self.eos_idx = 2

        self.device = 'cpu'

       # ## Shared tokenization transformation. Could evolve and make this unique to individuals in future##
        xlmr_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
        xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"

        self.tokenize = get_tokenizer("basic_english")
        self.vocab_transform = transforms.VocabTransform(load_state_dict_from_url(xlmr_vocab_path))
        self.truncate = transforms.Truncate(max_seq-2)
        self.pad = transforms.PadTransform(max_seq, 0)
        self.append_token = lambda idx, begin: transforms.AddToken(token=idx, begin=begin)

        self.shuffle = shuffle
        self.seed = 42

    def loadData(self, file_name, train, label, cols=["reviewText", "overall", "Category"], *add):
        dset = NLPDataset(file_name, train, label, cols=cols, *add)
        dataset_size = len(dset)
        indices = list(range(dataset_size))
        test_split = int(floor(0.8 * dataset_size))
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        # Split for Train and Test
        train_indices, test_indices = indices[test_split:], indices[:test_split]
        test_sampler = SubsetRandomSampler(test_indices)

        # Split for Train and Val
        val_split = int(floor(0.6 * len(train_indices)))
        train_indices, val_indices = train_indices[val_split:], train_indices[:val_split]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(dset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False, sampler=train_sampler)
        self.val_loader = DataLoader(dset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False, sampler=val_sampler)
        self.test_loader = DataLoader(dset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False, sampler=test_sampler)

    def collate_fn(self, inp):
        x_mod = pad_sequence(list(map(lambda x_new: self.transformText(x_new[0]), inp)))
        y_mod = torch.Tensor(list(map(lambda x_new: x_new[1], inp)))
        x_mod = torch.transpose(x_mod, 1, 0)
        return x_mod.to(self.device), y_mod.to(self.device)
        
    def transformText(self, text):
        try:
            out = self.tokenize(text)
        except:
            print(text)
            raise Exception
        out = self.vocab_transform(out)
        out = self.truncate(out)
        out = self.append_token(idx=0, begin=True)(out)
        out = self.append_token(idx=2, begin=False)(out)
        out = torch.tensor(out)
        out = self.pad(out)
        return out

