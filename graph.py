import numpy as np
import os
import sys
import pickle

with open('./data/adj_mx_3d.pkl', 'rb') as pf:
    adj_mx_data = pickle.load(pf)

adj_mx = adj_mx_data[2]
adj_mx.shape
print(adj_mx)