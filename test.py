from main import Engine

engine = Engine()
tree, execable = engine.generateSampleIndividual()
print(tree)
print(execable(10))

