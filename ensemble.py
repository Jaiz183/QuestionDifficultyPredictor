import random 
import numpy as np 
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)
from sklearn.impute import KNNImputer
from knn import knn_impute_by_user
from item_response import *

def bagging(matrix, n):
    """
    Given a matrix where "points" are rows and features are columns,
    return n matrices produced by sampling from the origin matrix with replacement.
    """
    height, width = matrix.shape
    # For each step, randomly generate a number from 0 to height-1, corresponding to index of the row of `matrix`
    # that we are going to sample
    ensemble_indices =  np.array([[random.randint(0, height-1) for _ in range(height)] for _ in range(n)])
    # `ensemble_indices` is an array of shape (n, 524), where the each row includes the indices of the rows in the 
    # original matrix `matrix` that were sampled.
    # Using advanced numpy indexing, for each of the n rows (each row corresponding to a bag) we 
    # extract the required rows from `matrix`. 
    ensemble = np.array([matrix[inds] for inds in ensemble_indices])
    return ensemble

class Learner:
    """
    A wrapper for a learner/imputer.

    The reason is that we wanted to try bagging for several different base models, and this provides 
    a consistent way (a contract) to do this.

    Attributes:
        - name: the name of the learner (str)
        - hyper: a dictionary with any hyperparameters needed for the model (dict)
    """

    def __init__(self, name, hyper):
        self.name = name
        self.hyper = hyper

    def learn(self, matrix):
        """
        Given the sparse matrix of students as rows and questions as columns, 
        and using the dictionary attribute of hyperparameters `self.hyper`,
        return a matrix with the Nans filled.
        """
        raise NotImplementedError


class KNNLearner(Learner):
    """
    Learner implementation of user-based KNN
    """
    def __init__(self, k, name="KNN"):
        hyper = {"k": k}
        super().__init__(name=name, hyper=hyper)
    
    def learn(self, matrix):
        nbrs = KNNImputer(n_neighbors=self.hyper['k'])
        mat = nbrs.fit_transform(matrix)
        return mat


def ensemble_predict(ensemble, learner, display=False):
    pred_mats = np.array([None for _ in range(3)])

    for i in range(len(ensemble)):
        tsamp = ensemble[i]
        mat = learner.learn(tsamp)
        pred_mats[i] = mat
        if display:
            singular_acc = sparse_matrix_evaluate(test_data, mat)
            print(f"singular_acc: {singular_acc}")
        
    stacked = np.stack(pred_mats, axis=2)
    avg = np.mean(stacked, axis=2)
    avg_rounded = avg.round(decimals=0)
    
    return avg_rounded


if __name__ == "__main__":
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    ensemble = bagging(sparse_matrix, 3)
    
    # ---- KNN ----
    k = 8
    knn_learner = KNNLearner(k=k)

    # ---- Prob ----
    ...

    # ---- NN ----
    ...

    final_mat = ensemble_predict(ensemble, knn_learner, display=True)
    acc = sparse_matrix_evaluate(test_data, final_mat)
    print(f"ensemble_acc: {acc}")
    print(f"original_acc: {sparse_matrix_evaluate(test_data, knn_learner.learn(sparse_matrix))}")    

