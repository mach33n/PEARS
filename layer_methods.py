from enum import Enum
from torch import nn, optim

# Custom classes used for restricting typed nature of GP
LayerList = type("LayerList", (list,), {'content': {}})
LayerList_Activation = type("LayerList_Activation", (object,), {'content': {}})

NumLayers = type("NumLayers", (int,), {'content': {}})
Dropout = type("Dropout", (float,), {'content': {}})

# Two custom classes to restrict primitive connections.
# Will want to go back and add custom input/output classes
# that account for train, val, test, weights, scores, etc.
Model_Input = type("Model_Input", (list,), {'content': {}})
Model_Output = type("Model_Output", (list,), {'content': {}})

## Enums for hyperparameter search space
class Optimizer(Enum):
    ADAM = optim.Adam

class Loss(Enum):
    MSE = nn.MSELoss

def cNetInd(layer_list, optimizer, loss):
    network = nn.Sequential(*layer_list)
    print(network)
    print("Compiler")
    print(optimizer)
    print(layer_list)
    print(loss)
    return "Done"

def Input():
    return []

""" Regular primitive wrapper for LSTMLayer from Pytorch. Consider
    adding projection specific variant. https://arxiv.org/abs/1402.1128"""
def LSTMLayer(layer_list, hidden_size, num_layers, dropout):
    layer_list.append(nn.LSTM(256, hidden_size, num_layers, dropout=dropout))
    return layer_list

justTerminals = [int, NumLayers, Dropout, Optimizer, Loss]
