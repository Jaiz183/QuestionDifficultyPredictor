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

# random.seed(1)
random.seed(8)
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

    def __init__(self, model, name, hyper):
        self.model = model
        self.name = name
        self.hyper = hyper
        self.fitted = False

    def fit(self, matrix):
        """
        Given the sparse matrix of students as rows and questions as columns, 
        and using the dictionary attribute of hyperparameters `self.hyper`,
        fit the model.

        This should be run before self.predict is called -- self.model
        will be fit and then predictions can be made.

        self.fitted should be set to True in the end of this function
        """
        raise NotImplementedError

    def predict(self, matrix):
        """
        Given a sparse matrix (of students as rows and questions as columns)
        predict missing values based on the fit model.

        Assumes self.fit has already been called.
        """
        raise NotImplementedError


class KNNLearner(Learner):
    """
    Learner implementation of user-based KNN
    """
    def __init__(self, k, name="KNN"):
        hyper = {"k": k}
        model = KNNImputer(n_neighbors=k)
        super().__init__(name=name, model = model, hyper=hyper)
    
    def fit(self, matrix):
        self.model.fit(matrix)
        self.fitted = True
    
    def predict(self, matrix):
        if not self.fitted:
            raise BaseException("Call self.fit before self.predict!")
        return self.model.transform(matrix)

class PartBLearner(Learner):
    """
    Learner implementation of user-based KNN
    """
    def __init__(self, k, name="KNN"):
        hyper = {"k": k}
        model = KNNImputer(n_neighbors=k)
        super().__init__(name=name, model = model, hyper=hyper)
    
    def fit(self, matrix):
        self.model.fit(matrix)
        self.fitted = True
    
    def predict(self, matrix):
        if not self.fitted:
            raise BaseException("Call self.fit before self.predict!")
        return self.model.transform(matrix)


def ensemble_predict(orig_matrix, ensemble, learner, display=False):
    n = len(ensemble)
    height, width = orig_matrix.shape
    pred_mats = np.zeros((n, height, width))

    # For each bag in the ensemble, fit it,
    # make the imputations and accumulate them,
    # so that they can then be averaged out
    # and yield the final prediction
    for i in range(len(ensemble)):
        bag = ensemble[i]
        learner.fit(bag)
        mat = learner.predict(orig_matrix)
        pred_mats[i] = mat
        if display:
            singular_acc_test = sparse_matrix_evaluate(test_data, mat)
            singular_acc_valid = sparse_matrix_evaluate(val_data, mat)
            print(f"singular_acc: {singular_acc_test, singular_acc_valid}")
    
    avg = np.mean(pred_mats, axis=0)
    return avg


if __name__ == "__main__":
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    ensemble = bagging(sparse_matrix, 3)
    
    # ---- KNN ----
    k = 11
    knn_learner = KNNLearner(k=k)

    # ---- Prob ----
    ...

    # ---- NN ----
    ...

    print("---------------- KNN ---------------- ")
    final_mat = ensemble_predict(sparse_matrix, ensemble, knn_learner, display=True)
    acc_test = sparse_matrix_evaluate(test_data, final_mat)
    acc_valid = sparse_matrix_evaluate(val_data, final_mat)
    print(f"ensemble_acc: {acc_test, acc_valid}")
    knn_learner.fit(sparse_matrix)
    orig_mat_res = knn_learner.predict(sparse_matrix)
    print(f"original_acc: {sparse_matrix_evaluate(test_data, orig_mat_res)}")    
    print("------------------------------------ ")

    print("---------------- NeuralNet ---------------- ")
    print("------------------------------------------- ")

    print("---------------- Probalisitic ---------------- ")
    print("---------------------------------------------- ")
