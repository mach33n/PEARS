from main import Engine
import weakref

# Play with generating individuals
#engine = Engine()
#pool = engine.generateNetInds(count=50)
#for i in pool:
    # deap enables you to print out interpretable function 
    # strings representing networks
    #print(str(i))

# Run simple search without metrics or with
# Bug: Unable to make multiple engines at once
metrics = {"empty": (1.0, lambda preds, truths: 1)}

engine = Engine(metrics=metrics)
engine.search()

