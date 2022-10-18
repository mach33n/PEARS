import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torchtext.transforms as transforms
from torch.hub import load_state_dict_from_url

import pandas as pd

from math import floor, ceil

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class NLPDataset(Dataset):
    def __init__(self, file_name):
        nlp_df = pd.read_csv(file_name, usecols=["reviewText", "overall", "Category"], dtype={'reviewText': 'str', 'overall':'int', 'Category':'str'})
        print(nlp_df.shape[0])
        nlp_df = nlp_df.dropna(subset=['reviewText', 'overall'])
        print(nlp_df.shape[0])
        self.x_train = nlp_df["reviewText"]
        self.y_train = nlp_df["overall"]
        categories = nlp_df["Category"]
        print(self.x_train.shape[0])
        print(self.y_train.shape[0])

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx], self.y_train[idx]


class AmazonDataLoader:
    def __init__(self, batch_size=16, max_seq=256, shuffle=True):
        ## Custom Parameters
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.bos_idx = 0
        self.pad_idx = 1
        self.eos_idx = 2

       # ## Shared tokenization transformation. Could evolve and make this unique to individuals in future##
        xlmr_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
        xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"

        self.tokenize = get_tokenizer("basic_english")
        self.vocab_transform = transforms.VocabTransform(load_state_dict_from_url(xlmr_vocab_path))
        self.truncate = transforms.Truncate(max_seq-2)
        self.pad = transforms.PadTransform(max_seq, 0)
        self.append_token = lambda idx, begin: transforms.AddToken(token=idx, begin=begin)

    def loadData(self, file_name):
        dset = NLPDataset(file_name)
        tset_ratio = floor(len(dset) * 0.8)
        train_set, test_set = random_split(dset, [tset_ratio, len(dset) - tset_ratio])

        vset_ratio = floor(len(train_set) * 0.6)
        train_set, val_set = random_split(train_set, [vset_ratio, len(train_set)-vset_ratio])

        self.train_loader = DataLoader(train_set, batch_size=10, collate_fn=self.collate_fn, shuffle=False)
        self.val_loader = DataLoader(val_set, batch_size=10, collate_fn=self.collate_fn, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=10, collate_fn=self.collate_fn, shuffle=False)

    def collate_fn(self, inp):
        x_mod = pad_sequence(list(map(lambda x_new: self.transformText(x_new[0]), inp)))
        y_mod = torch.Tensor(list(map(lambda x_new: x_new[1], inp)))
        #print("Here")
        #print(x_mod[1])
        #print(type(x_mod))
        #print(y_mod.shape)
        x_mod = torch.transpose(x_mod, 1, 0)
        return x_mod.to(device), y_mod.to(device)
        
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

class SampleSentAnalyser(nn.Module):
    def __init__(self):
        super(SampleSentAnalyser, self).__init__()
        self.dense = nn.LazyLinear(30)
        #self.LSTM = nn.LSTM(256, 20, 10)
        self.classify = nn.LazyLinear(1)

    def forward(self, x):
        inp = x
        #print(x.shape)
        #x, (hidden, cell) = self.LSTM(x)
        x = x.float()
        x = self.dense(x)
        x = self.classify(x)
        out = x
        return out

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model = model.to(device)
    criterion = criterion.to(device)

    model.train()

    for idx, (tokens, labels) in enumerate(iterator):
        print(idx)
        optimizer.zero_grad()
        
        predictions = model(tokens).squeeze(1)
        
        loss = criterion(predictions, labels)
        
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


criterion = nn.BCEWithLogitsLoss()

loader_obj = AmazonDataLoader()
loader_obj.loadData("data/amazon_dataset_preprocessed_30k.csv")

model = SampleSentAnalyser()

optimizer = optim.Adam(model.parameters())

train(model, loader_obj.train_loader, optimizer, criterion)
