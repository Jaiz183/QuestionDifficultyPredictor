import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)

    ### NOTE
    # In `knn_impute_by_user`, we assumed that a student's answer to a question
    # is expected to be similar to the answers of "similar" students to that 
    # question (where similarity is measured as a Nan-tolerant euclidean distance 
    # between the answer vectors of students)
    #
    # Here, where we perform knn based on question similarity, the underlying assumption is different. 
    # We assume that if two questions have the same answers from students (that is, is students 
    # that answered correctly one question also answered the other question correctly, and 
    # students that answered one question falsely also answered the other question falsely), 
    # then the answer of some student to the first question should be the same as the 
    # answer of the SAME student in the other question.
    ###

    ### NOTE: We assume the inputted matrix has students as rows and questions as columns.
    # The reason in a matrix, a "point" is conceived to be a row (at least that's how 
    # scikit treats matrices; the array [[1,2], [4,2], [4,9], [1,10]] has 4 points of 2 
    # dimensions/features each). 
    # Assuming it hasn't been mutated before entering this function in any way, the matrix
    # will have one row for each student and one column for each question. Thus, simply 
    # taking the transpose of the matrix gives a row for each question and a column for each 
    # student.
    # Note how the representation changes. Now each "point" in space is a questions, with 
    # N-dimensions, N being the number of students. For each student, we have whether 
    # they answered that question correctly or not.
    mat_trans = matrix.T

    # We use NaN-Euclidean distance measure.
    imputed_mat_trans = nbrs.fit_transform(mat_trans)

    # Not that the `sparse_matrix_evaluate` function expects a matrix with shape
    # (num students)-by-(num-questions), so we take the transpose to return 
    # to the original representation.
    imputed_mat = imputed_mat_trans.T
    acc = sparse_matrix_evaluate(valid_data, imputed_mat)
    print("Validation Accuracy: {}".format(acc))
    return acc
    acc = None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc

def plot(xs, ys, genlabel, xlabel, ylabel, title, show=True, save=False, filepath="plot.png"):
    plt.figure(figsize=(10,6))
    plt.scatter(xs, ys, label = genlabel )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig(filepath) # must be run from within `~/starter` directory

def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    knn_func = knn_impute_by_item 
    # knn_func = knn_impute_by_user

    k_vals = [1, 6, 11, 16, 21, 26]
    accs = np.array([knn_func(sparse_matrix, val_data, k) for k in k_vals])
    k_best = np.argmax(accs)

    filename = "knn-item.png" if knn_func == knn_impute_by_item else "knn-stud.png"

    plot(xs=k_vals, ys=accs, genlabel="k-vs-acc", xlabel="k", ylabel="validation-accuracy", 
            title="Validation Accuracy for different k", save=True, show=False, filepath=f"./knn-results/{filename}")
    
    test_acc = knn_func(sparse_matrix, test_data, k_vals[k_best])
    best_k_msg = f"The highest performing k is k*={k_vals[k_best]}, with test accuracy {test_acc}"
    
    print(best_k_msg)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
