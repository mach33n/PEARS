from torch import cuda, device
import torch.nn as nn

# Temporarily ignoring option to build and just relative importing deap right now
from deap.deap import base, creator, gp, tools

import data_loader as DL
import layer_methods as LM
import random

class Engine:
    def __init__(self, batch_size=16, max_seq=256):
        self.toolbox = base.Toolbox()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.dataloader = DL.EngineDataLoader(batch_size=batch_size, max_seq=max_seq, shuffle=True)

        ## DEAP Specific
        # default assumes accuracy, latency
        self.max_gens = 1000
        self.fitness_weights = (1.0,-1.0)
        self.min_size = 3
        self.max_size = 10
        self.pop_size = 100


    """ Primary method for running evolutionary search."""
    def run(self):
        self.setupDEAPTools()
        self.search()


    """ Generate custom classes and add useful
        methods to toolbox """
    def setupDEAPTools(self):
        # Define custom classes used by deap
        creator.create("Fitness", base.Fitness, weights=self.fitness_weights)
        creator.create("NetInd",
                list,
                fitness=creator.Fitness,
                age=0,
                elapsed_time=0,
                parents=None
        )
        creator.create("population",
                list,
                num_gens=0,
                evaluated_inds=0,
                unevaluated_inds=0
                )

        # Add custom content to toolbox
        self.pset = gp.PrimitiveSetTyped("NetIndPSet", [], LM.Model_Output) 
        self.genPSET()

        self.toolbox.register("instNetInd", gp.genNetInd, self.pset, self.max_size)
        self.toolbox.register("individual", tools.initIterate, creator.NetInd, self.toolbox.instNetInd)
        self.toolbox.register("population", tools.initRepeat, creator.population, self.toolbox.individual, self.pop_size)

        self.population = self.toolbox.population()
        
        # Sample instantiation of Individual(Debug)
        #expr = gp.genNetInd(self.pset, 7)
        #expr = self.toolbox.individual()
        #print(expr)
        #tree = gp.PrimitiveTree(expr)
        #individual = creator.NetInd([tree])
        #print()
        #print(tree)
        #print()
        #print(individual)

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

    def search(self):
        gen = 1
        while gen < self.max_gens:
            print("Generation " + str(gen))
            for ind in self.population:
                # Do evaluation
                #print(ind)
                # Set new fitness values on individual
                ind.fitness.values = (random.uniform(0,1), random.uniform(0,1))
            # Do selection
            # There is variability in selection method but generally NSGA2 isn't bad
            self.population = tools.selNSGA2(self.population, 20)
            # Do mate and mutate
            # Put this off until future

            gen += 1

engine = Engine()
engine.run()
