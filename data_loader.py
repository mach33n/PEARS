from torch.utils.data import Dataset, DataLoader
import torchtext.transforms as transforms
from torch.hub import load_state_dict_from_url

class EngineDataLoader:
    def __init__(self, batch_size=16, max_seq=256, shuffle=True):
        ## Custom Parameters
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.bos_idx = 0
        self.pad_idx = 1
        self.eos_idx = 2

        ## Using pytorch provided data for now, might want to incorporate custom later. ##
       # self.train_data = SST2(split="train")
       # self.val_data = SST2(split="dev")
       # self.test_data = SST2(split="test")

       # self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
       # self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
       # self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

       # ## Shared tokenization transformation. Could evolve and make this unique to individuals in future##
       # xlmr_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
       # xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"

       # self.tokenize = transforms.SentencePieceTokenizer(xlmr_model_path)
       # self.vocab_transform = transforms.VocabTransform(load_state_dict_from_url(xlmr_vocab_path))
       # self.truncate = transforms.Truncate(max_seq)
       # self.append_token = lambda idx, begin: transforms.AddToken(token=idx, begin=begin)
        

    def collate_fn(self, inp):
        return list(map(lambda x: self.transformText(x[0]), inp)), list(map(lambda x: x[1], inp))
        
    def transformText(self, text):
        out = self.tokenize(text)
        out = self.vocab_transform(out)
        out = self.truncate(out)
        out = self.append_token(idx=0, begin=True)(out)
        out = self.append_token(idx=2, begin=False)(out)
        return out
        
    # Dev Helper functions
    def outputText(self):
        for t_id, label in self.train_loader:
            print(t_id)
            print(label)
            print()
            break
        
