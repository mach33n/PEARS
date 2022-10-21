from torch import cuda, device
import torch.nn as nn

# Temporarily ignoring option to build and just relative importing deap right now
from deap.deap import base, creator, gp, tools

from data_loader import AmazonDataLoader
from layer_methods import PSetHub
import random
import pickle
from copy import deepcopy

class Engine:
    def __init__(self, max_gens=50, metrics=None, batch_size=16, max_seq=256):
        self.toolbox = base.Toolbox()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.dataloader = AmazonDataLoader()
        self.psetH = PSetHub()

        ## DEAP Specific
        # default assumes accuracy, latency
        self.max_gens = max_gens
        self.metrics = metrics
        weights = [1.0]
        if self.metrics != None:
            for name in self.metrics:
                weights.append(self.metrics[name][0])
        print(tuple(weights))
        self.fitness_weights = tuple(weights)
        self.min_size = 3
        self.max_size = 10
        self.init_size = 25
        self.pareto_size = 20

        self.setupDEAPTools()

    """ Primary method for running evolutionary search."""
    def run(self):
        self.search()

    """ Generate custom classes and add useful
        methods to toolbox """
    def setupDEAPTools(self):
        # Define custom classes used by deap
        creator.create("Fitness", base.Fitness, weights=self.fitness_weights)
        creator.create("NetInd",
                list,
                fitness=creator.Fitness,
                age=1,
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
        self.toolbox.register("individual", tools.initIterate, creator.NetInd, self.generateNetInds)
        self.toolbox.register("population", tools.initRepeat, creator.population, self.toolbox.individual, self.init_size)
        self.toolbox.register("refresh_population", tools.initRepeat, creator.population, self.toolbox.individual, self.init_size-self.pareto_size)


    def search(self):
        self.population = self.toolbox.population()
        print(self.population)

        gen = 1
        while gen < self.max_gens:
            print("Generation " + str(self.population.num_gens))
            for idx in range(len(self.population)):
                ind = deepcopy(self.population[idx])
                # Do evaluation
                execable = self.psetH.compile(ind[0], self.psetH.pset())
                loss, accuracy, metOuts = execable(self.metrics)

                # Set new fitness values on individual
                vals = [accuracy]
                for metName in metOuts:
                    vals.append(metOuts[metName])
                ind.fitness.values = tuple(vals)

                self.population[idx] = ind

            # Do selection
            # There is variability in selection method but generally NSGA2 isn't bad
            print(tools.selNSGA2(self.population, self.pareto_size))
            print(len(tools.selNSGA2(self.population, self.pareto_size)))

            pareto_front = deepcopy(tools.selNSGA2(self.population, self.pareto_size))
            self.population.clear()
            self.population.extend(pareto_front)

            self.population.extend(self.toolbox.refresh_population())
            print(len(self.population))

            self.logParetoOptimalInds(self.population, self.population.num_gens, verbose=True)
            
            # Do mate and mutate

            self.population.num_gens += 1

    # Need to generate log file for storing pandas df full of pareto front information 
    # based on generation recorded
    def logParetoOptimalInds(self, paretoFront, generation, verbose=False):
       pass 

    def generateNetInds(self, count=1):
        # Sample instantiation of Individual
        pool = []
        badInits = 0
        while len(pool) < count:
            try:
                tree = self.psetH.genNetInd(self.psetH, 2, maxAttempts=20, debug=True)
                pool.append(tree)
            except Exception as e:
                badInits += 1
                if badInits > 2*count:
                    print("Consider reconfiguring primitive set. The generator is struggling to piece together valid architectures for you. Expect a slight performance slowdown.")
                    badInits = 0
                
        return pool

#metrics = {"empty": (1.0, lambda preds, truths: 1)}
#engine = Engine()
#engine = Engine(metrics=metrics)
#engine.search()
