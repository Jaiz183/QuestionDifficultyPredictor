import ast
import csv
import os

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import KNNImputer
from ensemble import KNNLearner
from utils import load_train_sparse, sparse_matrix_evaluate, load_valid_csv, \
    load_public_test_csv

from scipy.cluster.hierarchy import DisjointSet

array = np.reshape([1, 2, 3, 4, 5, 6], shape=(2, 3))
print(array[[1, 0]])
