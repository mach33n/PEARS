# EALib
Library of developing modules and classes to run and experiment with evolutionary algorithms.

Link to Trello Board: https://trello.com/invite/b/SMybqEcB/6bb8065253c50eb299366f92d209ce76/ealib-dev


## Overview of Files:
# Main.py
  - Engine
    - generateNetInds(counts=1)
      -This method can be utilized for debugging purposes to play around with custom amounts of individuals generated from the psetHub available to the engine. Additionally, this method is used in the search method for generating new individuals for search space.
    - logParetoOptimalInds: TBD
    - run - just calls search for now, ideally will be used to execute pre and post methods specific to search configuration specified
    - search
      -Utilizes max_gen variable to run some n number of iterations in which individuals are generated, evaluated and selected using NSGA2
    - setupDEAPTools
      -Initializes dynamic classes such as NetInd, Population and Fitness. Additionally, registers iterative methods that serve as shorthands for instantiation of populations, individuals, etc.

# Data_Loader.py
  - NLPDataset
    - Reads specified CSV and extracts columns specific for getting x and y data.
  - AmazonDataLoader
    - Currently uses an xlmr vocab transformer and basic tokenizer to initialize train, val and test data specific to sentiment analysis training.
    - In theory, this loader could be more effectively expanded to handling larger variety of datasources. Currently crunched for time so some values are hardcoded.

# Layer_Methods.py  
  - PSetHub
    - buildPSet
     - initializes deap primitive set typed and all available layers for the search space.
    - Primitives
     - Most methods in this section are simply wrappers to existing pytorch methods that accpet and configure specifc values that are exposed through the PSET. One complex issue currently holding us back is the question of how to validate layer combinations that maintain legal output dimensions. Expecting to implement another generator specific to layer lists. 
    - Terminals
     - Currently only the input layer is a terminal requiring a function. Can't add it as a terminal in the other way because lists are not hashable and thus cause errors in deap(idrk). Could be something to look into. 
    - Modified DEAP Methods
     - GenMethods
      - These two methods are responsible for how deap initializes individuals. It essentially works backwards from the expected output of our pset until it's reached a certain depth or terminal type. One thing this method does not do which it has been modified to do is recognize when no primitives exist for a type and just pull a terminal. This and other reasons are why we will keep a modified generator method outside of the deap folder for usage. 
     - Compile
      - This method is slightly modified to add a check parameter to the compile method in the event that we want to validate neural networks without training them(in the generator method).
    - Custom Classes
     - These custom classes are recognized at surface level by deap as a means of validating certain connections. Alot of deap's generative methodology is completely random thus we have to put in alot of effort to manually restrict what can be put together.   
