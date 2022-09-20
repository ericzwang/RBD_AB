import torch
from torch import nn
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
import gc
import sys
import matplotlib.pyplot as plt
import random
import os
from NeuralNetwork import NeuralNetwork
from processing_functions import *
from scipy.stats import spearmanr
from copy import deepcopy


if __name__ == "__main__":
    
    test_type = sys.argv[1]
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chains = ['heavy_variable', 'light_variable', 'rbd']
    
    model = NeuralNetwork(201, 21, 36, 6, 1, 32, 64, 256, 32)
    model.load_state_dict(torch.load('parameters.pt', map_location=device), strict=False)
    model.to(device)
    
    if test_type == 'reincke':
        train_data = pd.read_csv('data/train.csv')
        df = getreinckedf(chains, train_data)
    else:
        df = pd.read_csv(f'data/{test_type}.csv')
        
    predictions = []
    for trial in range(100):
        gc.collect(); torch.cuda.empty_cache()
        prediction, escapes = getprediction(df, chains, 0.75, device, model)
        predictions.append(prediction)

    prediction = np.mean(predictions, axis=0)
    pear, spear = np.corrcoef(prediction, escapes)[0,1], spearmanr(prediction, escapes).correlation

    #bootstrapping the standard error
    combine = np.vstack([prediction, escapes]).T
    pearse = bootstrap_se(combine, np.corrcoef, lambda x: x[0,1])
    spearse = bootstrap_se(combine, spearmanr, lambda x: x.correlation)
    print(f"Pearson:{pear:.3f}+-{pearse:.3f} Spearman:{spear:.3f}+-{spearse:.3f}")
        
