import pandas as pd
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pickle
import copy
import os
import pickle
import csv
import sys




# ptdf = sci.loadmat (address + 'ptdf' )['ptdf']
# w = ptdf[0][0]

# ram = sci.loadmat (address +'ram' )['ram']
# x = ram[0][0]


# sampling = sci.loadmat (address +'sampling' )['sampling']
# y = sampling[0][1]

with open('ptdf.pickle', 'rb') as f:
    ptdf = pickle.load(f)
    
with open('ram.pickle', 'rb') as f:
    ram = pickle.load(f)
    
with open('sampling.pickle', 'rb') as f:
    sampling = pickle.load(f)


nbPoints = len(sampling[0])
#matrix = false(nbPoints,nbPoints,nbPoints,nbPoints);
d = 4
matrix = np.zeros(shape=[nbPoints]*d)

nhubs = ptdf[0][0][0].size
netHubPositions = np.zeros((nhubs, 1))

for i in range(nbPoints):
    netHubPositions[0] = sampling[0][i-1]
    for j in range(nbPoints):
        netHubPositions[1] = sampling[0][j-1]
        for n in range(nbPoints):
            netHubPositions[2] = sampling[0][n-1]
            tmp = -sampling[0][i-1]-sampling[0][j-1]-sampling[0][n-1]
            for m in range(nbPoints):
                print(i,j,n,m)
                netHubPositions[3] = sampling[0][m-1]
                netHubPositions[4] = tmp -sampling[0][m-1];
                if all(np.dot (ptdf[0][0], netHubPositions)<= ram[0][0]):
                    matrix[i,j,n,m] = True
                else:
                    matrix[i,j,n,m] = False


# with open('ptdf.pickle', 'wb') as f:
#     pickle.dump(ptdf, f)
    
# with open('ram.pickle', 'wb') as f:
#     pickle.dump(ram, f)
    
# with open('sampling.pickle', 'wb') as f:
#     pickle.dump(sampling, f)
    
