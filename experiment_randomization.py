import numpy as np
import random

obj_classes = ['Apple', 'Cucumber', 'Carrot']
obj_sizes = ['1', '1/2', '1/4', '1/8']
heuristics = ['even', 'long']
n_objects = random.randint(1,10)


exp_dict = {}
for i in range(n_objects):
    size = obj_sizes[random.randint(0,3)]
    obj = obj_classes[random.randint(0,2)]
    exp_dict[i+1] = (obj, size)
print("\nExperiment: ", exp_dict)
print("\nHeuristic: ", heuristics[random.randint(0,1)])