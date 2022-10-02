from torch import cuda, device
from torch.hub import load_state_dict_from_url
from torchtext.datasets import SST2
import torchtext.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Temporarily ignoring option to build and just relative importing deap right now
from deap.deap import base, creator, gp, tools

import layer_methods as LM
import random

class Engine:
    def __init__(self, batch_size=16, max_seq=256):
        self.toolbox = base.Toolbox()
        self.device = device("cuda" if cuda.is_available() else "cpu")

        ## Custom Parameters
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.bos_idx = 0
        self.pad_idx = 1
        self.eos_idx = 2

        ## DEAP Specific
        # default assumes accuracy, latency
        self.fitness_weights = (1.0,-1.0)
        self.min_size = 3
        self.max_size = 10

        ## Using pytorch provided data for now, might want to incorporate custom later. ##
        self.train_data = SST2(split="train")
        self.val_data = SST2(split="dev")
        self.test_data = SST2(split="test")

        #print(self.transformBatch(self.train_data))

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        ## Shared tokenization transformation. Could evolve and make this unique to individuals in future##
        xlmr_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
        xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"

        self.tokenize = transforms.SentencePieceTokenizer(xlmr_model_path)
        self.vocab_transform = transforms.VocabTransform(load_state_dict_from_url(xlmr_vocab_path))
        self.truncate = transforms.Truncate(max_seq)
        self.append_token = lambda idx, begin: transforms.AddToken(token=idx, begin=begin)

    """ Primary method for running evolutionary search."""
    def run(self):
        #self.outputText()
        self.setupDEAPTools()

    """ Generate custom classes and add useful
        methods to toolbox """
    def setupDEAPTools(self):
        # Define custom classes used by deap
        creator.create("Fitness", base.Fitness, weights=self.fitness_weights)
        creator.create("NetInd",
                list,
                fitness=creator,
                age=0,
                elapsed_time=0,
                parents=None
        )

        # Add custom content to toolbox
        self.pset = gp.PrimitiveSetTyped("NetIndPSet", [], LM.Model_Output) 
        self.genPSET()

        self.toolbox.register("instNetInd", gp.genNetInd, self.pset, self.min_size, self.max_size)
        createNetInd = [self.toolbox.instNetInd]
        self.toolbox.register("individual", tools.initIterate, creator.NetInd, createNetInd)

        expr = gp.genNetInd(self.pset, 7)
        print(expr)
        tree = gp.PrimitiveTree(expr)
        individual = creator.NetInd([tree])
        print()
        print(tree)
        print()
        print(individual)

    def genPSET(self):
        # All available primitives
        self.pset.addPrimitive(LM.cNetInd, [LM.LayerList, LM.Optimizer, LM.Loss], LM.Model_Output)
        self.pset.addPrimitive(LM.LSTMLayer, [LM.EmptyLayerList, int, int, LM.NumLayers, LM.Dropout], LM.LayerList)
        self.pset.addPrimitive(LM.LSTMLayer, [LM.EmptyLayerList, int, int, LM.NumLayers, LM.Dropout], LM.LayerList_Activation)
        self.pset.addPrimitive(LM.Input, [], LM.EmptyLayerList)

        # All available terminals
        self.pset.addTerminal(256, int)
        self.pset.addTerminal(LM.Optimizer.ADAM, LM.Optimizer)
        self.pset.addTerminal(LM.Loss.MSE, LM.Loss)

        # Ephemeral Constants are simply ranges of values associated with primitive types like ints
        self.pset.addEphemeralConstant("NumLayers", lambda : int(random.uniform(1,7)), LM.NumLayers)
        self.pset.addEphemeralConstant("Dropout", lambda : random.uniform(0,1), LM.Dropout)

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

engine = Engine()
engine.run()
