from copy import deepcopy
import random
from enum import Enum
from torch import nn, optim
from deap.deap import gp
from inspect import isclass
import sys
import torch
import uuid

import warnings
warnings.filterwarnings("ignore") 

from data_loader import AmazonDataLoader

class PSetHub:
    def __init__(self):
        self.custId = uuid.uuid1()
        _pset = self.buildPSet()
        self.pset = lambda : deepcopy(_pset)
        self.justTerminals = [HiddenSize, int, NumLayers, Dropout, Optimizer, Criterion, CustMetricsList, NumberHeads]

    def buildPSet(self):
        pset = gp.PrimitiveSetTyped("NetIndPSet_" + str(self.custId), [CustMetricsList], Model_Output) 

        # All available primitives
        pset.addPrimitive(self.cNetInd, [LayerList, Optimizer, Criterion, CustMetricsList], Model_Output)
        pset.addPrimitive(self.LSTMLayer, [LayerList, HiddenSize, NumLayers, Dropout], LayerList)
        pset.addPrimitive(self.GRULayer, [LayerList, HiddenSize, NumLayers, Dropout], LayerList)
        pset.addPrimitive(self.TransformerEncoderLayer, [LayerList, NumberHeads, Dropout], LayerList)
        pset.addPrimitive(self.RNNLayer, [LayerList, NumberHeads, Dropout], LayerList)

        # ALL AVAILABLE TERMINALS #
        pset.addTerminal(self.Input(), LayerList)
        pset.addTerminal({"optimizer": optim.Adam}, Optimizer, "ADAM")
        pset.addTerminal({"criterion": nn.MSELoss}, Criterion, "MSELoss")


        # Ephemeral Constants are simply ranges of values associated with primitive types like ints
        pset.addEphemeralConstant("NumLayers", lambda : random.randint(1,7), NumLayers)
        pset.addEphemeralConstant("HiddenSize", lambda : random.randint(10,50), HiddenSize)
        pset.addEphemeralConstant("Dropout", lambda : random.uniform(0,1), Dropout)
        pset.addEphemeralConstant("GenericInt", lambda : random.randint(1,10), int)
        pset.addEphemeralConstant("GenericFloat", lambda : random.uniform(1,10), float)
        pset.addEphemeralConstant("GenericBool", lambda : random.choice([True, False]), bool)
        pset.addEphemeralConstant("NumberHeads", lambda: random.randint(1,2), NumberHeads)

        return pset

    ####################### Primitives ########################### 

    """ Regular primitive wrapper for LSTMLayer from Pytorch. Consider
        adding projection specific variant. https://arxiv.org/abs/1402.1128"""
    def LSTMLayer(self, layer_list, hidden_size, num_layers, dropout):
        class extract_tensor(nn.Module):
            def forward(self,x):
                return x[0]
        layer_list.append(nn.LSTM(random.choice([256]), hidden_size, num_layers, dropout=dropout))
        layer_list.append(extract_tensor())
        return layer_list

    def GRULayer(self, layer_list, hidden_size, num_layers, dropout):
        class extract_tensor(nn.Module):
            def forward(self,x):
                return x[0]
        layer_list.append(nn.GRU(random.choice([256]), hidden_size, num_layers, dropout=dropout))
        layer_list.append(extract_tensor())
        return layer_list

    def TransformerEncoderLayer(self, layer_list, nhead, dropout):
        layer_list.append(nn.TransformerEncoderLayer(random.choice([256,25]), nhead, dropout=dropout))
        return layer_list

    def RNNLayer(self, layer_list, hidden_size, num_layers, dropout):
        class extract_tensor(nn.Module):
            def forward(self, x):
                return x[0]
        layer_list.append(nn.RNN(random.choice([256]), hidden_size, num_layers, dropout=dropout))
        layer_list.append(extract_tensor())
        return layer_list

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc

    def cNetInd(self, layer_list, optimizer, criterion, cust_metrics=None, check=False):
        if not check:
            print("Compiling")
        dataloader = AmazonDataLoader()
        dataloader.loadData("data/amazon_dataset_preprocessed_30k.csv", "reviewText", "overall")
        
        #print(layer_list)
        layer_list.append(nn.LazyLinear(1))
        model = nn.Sequential(*layer_list)

        optimizer = optimizer['optimizer'](model.parameters())
        criterion = criterion['criterion']()

        if not check:
            print(model)
        #print(optimizer)
        #print(loss)

        epoch_loss = 0
        epoch_acc = 0

        #model = model.to(device)
        #criterion = criterion.to(device)

        model.train()
        if cust_metrics != None:
            custScores = dict.fromkeys(cust_metrics, 0)
        for idx, (tokens, labels) in enumerate(dataloader.train_loader):
            optimizer.zero_grad()
            print(model)
            # Necessary locally not sure why
            tokens = tokens.float() 
            predictions = model(tokens).squeeze(1)
            
            if cust_metrics != None:
                for met_name in cust_metrics:
                    custScores[met_name] += cust_metrics[met_name][1](predictions, labels)

            loss = criterion(predictions, labels)
            
            acc = self.binary_accuracy(predictions, labels)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            if check:
                return 
            
        if cust_metrics != None:
            for key in custScores:
                custScores[key] = custScores[key]/len(dataloader.train_loader)
        else:
            custScores = []
        return epoch_loss / len(dataloader.train_loader), epoch_acc / len(dataloader.train_loader), custScores

    ####################### Terminals ########################### 
    # Method for starting layerlist compilation
    def Input(self):
        return []

    ####################### Modified DEAP Methods ##############################
    """Custom Generator Method for restricting to only valid neural networks. """
    def genNetInd(self, psetH, max_, maxAttempts=25, debug=False):
        """Generate an expression where each leaf has the same depth
        between *min* and *max*.

        :param pset: Primitive set from which primitives are selected.
        :param max_: Maximum Height of the produced trees.
        :returns: A full tree with all leaves at the same depth.
        """

        def condition(height, depth, type_):
            """Expression generation stops when the depth is equal to height."""
            return depth >= height or type_ in psetH.justTerminals

        attempts = 0
        while attempts < maxAttempts:
            try:
                expr = self.generateNetInd(psetH.pset(), max_, condition)
                tree = gp.PrimitiveTree(expr)
                execable = self.compile(tree, pset=self.pset(), check=True)
                return tree
            except Exception as e:
                if debug:
                    print("BadInit")
                    print(e)
            attempts += 1
        if debug:
            print("Unable to find good individual")
        return None

    def generateNetInd(self, pset, max_, condition, type_=None):
        """ Custom Generator Function for restricting combinations of NetIndividuals 
        to structures that are valid and can be evaluated. See below gen function for 
        arg information. """
        if type_ is None:
            type_ = pset.ret
        expr = []
        height = max_
        stack = [(0, type_)]
        while len(stack) != 0:
            depth, type_ = stack.pop()
            if condition(height, depth, type_):
                try:
                    term = random.choice(pset.terminals[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "
                                      "a terminal of type '%s', but there is "
                                      "none availabl." % (type_,)).with_traceback(traceback)
                if isclass(term):
                    term = term()
                expr.append(term)
            else:
                try:
                    prim = random.choice(pset.primitives[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "
                                      "a primitive of type '%s', but there is "
                                      "none available." % (type_,)).with_traceback(traceback)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
        return expr

    def compile(self, expr, pset, check=False):
        """Compile the expression *expr*.

        :param expr: Expression to compile. It can either be a PrimitiveTree,
                     a string of Python code or any object that when
                     converted into string produced a valid Python code
                     expression.
        :param pset: Primitive set against which the expression is compile.
        :returns: a function if the primitive set has 1 or more arguments,
                  or return the results produced by evaluating the tree.
        """
        code = str(expr)
        if len(pset.arguments) > 0:
            # This section is a stripped version of the lambdify
            # function of SymPy 0.6.6.
            args = ",".join(arg for arg in pset.arguments)
            code = "lambda {args}: {code}".format(args=args, code=code)
        try:
            if check:
                code = code[:-1]
                code = code + ", check=True)"
                return eval(code, pset.context, {"check": True})
            else:
                return eval(code, pset.context, {"check": False})
        except MemoryError:
            _, _, traceback = sys.exc_info()
            raise MemoryError("DEAP : Error in tree evaluation :"
                                " Python cannot evaluate a tree higher than 90. "
                                "To avoid this problem, you should use bloat control on your "
                                "operators. See the DEAP documentation for more information. "
                                "DEAP will now abort.").with_traceback(traceback)

############################ Custom classes used for restricting typed nature of GP ############################
LayerList = type("LayerList", (list,), {'content': {}})
LayerList_Activation = type("LayerList_Activation", (object,), {'content': {}})

NumLayers = type("NumLayers", (int,), {'content': {}})
NumberHeads = type("NumberHeads", (int,), {'content': {}})
HiddenSize = type("HiddenSize", (int,), {'content': {}})
Dropout = type("Dropout", (float,), {'content': {}})
Optimizer = type("Optimizer", (object,), {'content': {}})
Criterion = type("Criterion", (object,), {'content': {}})
CustMetricsList = type("CustMetricsList", (dict,), {'content': {}})

# Two custom classes to restrict primitive connections.
# Will want to go back and add custom input/output classes
# that account for train, val, test, weights, scores, etc.
Model_Input = type("Model_Input", (list,), {'content': {}})
Model_Output = type("Model_Output", (list,), {'content': {}})
