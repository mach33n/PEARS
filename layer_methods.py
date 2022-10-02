from enum import Enum
from torch import nn, optim

# Custom classes used for restricting typed nature of GP
EmptyLayerList = type("LayerList", (list,), {'content': {}})
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

def cNetInd(layer_list, optimizer, loss_func):
    print("Compiler")
    pass

def Input():
    return []

""" Regular primitive wrapper for LSTMLayer from Pytorch. Consider
    adding projection specific variant. https://arxiv.org/abs/1402.1128"""
def LSTMLayer(layer_list, input_size, hidden_size, num_layers, dropout):
    layer_list.append(nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout))

justTerminals = [int, NumLayers, Dropout, Optimizer, Loss]

