
import numpy as np
from pysat.solvers import Glucose3
from collections import defaultdict
import time
import os
import torch

"""
used to plot figures

"""
def data_clean(filename):
    print(filename)
    data = torch.load(filename)
    print(len(data))
    #num_nodes
    num_nodes = [ele[0] for ele in data]
    num_edges = [ele[1] for ele in data]
    num_dofds = [ele[2] for ele in data]
    run_times = [ele[3] for ele in data]
    
    pass

if __name__ == '__main__':

    
    data_path = "outputs/original_graphs.pth"
    data_clean(data_path)
