import os
import csv
import ast
from pprint import pprint

from matplotlib import pyplot as plt

from utils import _load_csv, load_train_sparse, \
    load_valid_csv, sparse_matrix_evaluate, load_public_test_csv
import numpy as np
from sklearn.impute import KNNImputer
from ensemble import bagging

# manually seen form csv
NUM_SUBJ = 388


def load_meta(path):
    """
    Helper that transforms the question metadata csv file 
    (including the subjects related to each question)
    to a dictionary where the keys are question id's 
    and the values are the subject list of each question.
    """
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data -- key-value parts of the form qid:subjects
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data[int(row[0])] = ast.literal_eval(row[1])
            except ValueError:
                # print(f"Value Error: {row}")
                pass
            except IndexError:
                print(f"Index Error: {row}")
                pass
    return data


def compute_subject_matrix():
    """
    This takes us from the 542-by-1774 sparse matrix 
    of students and subjects to a 542-by-387 matrix
    of students and subjects, where each cell (i,j)
    represents the proportion of questions in 
    subject `j` that student `i` got correct 
    (out of all observed questions for that student)
    """
    metadata = load_meta("./data/question_meta.csv")
    sparse_matrix = load_train_sparse("./data").toarray()
    num_users, num_questions = sparse_matrix.shape

    subj_count = np.zeros((num_users, NUM_SUBJ))
    new_mat = np.zeros((num_users, NUM_SUBJ))

    for user_id in range(num_users):
        for question_id in range(num_questions):
            answer = sparse_matrix[user_id, question_id]
            if np.isnan(answer):
                continue
            for sub_id in metadata[question_id]:
                # If observed, add one to the count 
                # of subject `sub_id` for that students 
                # and also one to the count of correct 
                # answers in subject `sub_id` for that 
                # student if the student answered correctly.
                new_mat[user_id][sub_id] += answer
                subj_count[user_id, sub_id] += 1

    # If some student did not answer any questions 
    # for a given subject, we put nan to avoid 
    # division by 0 in the next step
    subj_count[subj_count == 0] = np.nan

    contributions = np.zeros(sparse_matrix.shape)
    # Compute each cell as described in the docstring
    for i in range(num_users):
        for j in range(num_questions):
            ans = sparse_matrix[i, j]
            if np.isnan(ans):
                continue
            contributions[i][j] += ans * len(metadata[j])
    assert np.nansum(new_mat) == np.nansum(contributions)

    # Right now `new_mat` has absolute counts so we divide 
    # by subj_count to get the desired proportions.
    final_mat = np.divide(new_mat, subj_count)

    # Need to do this for subjects for which not a 
    # single question was observed.
    for i in range(NUM_SUBJ):
        row = final_mat[:, i]
        if np.isnan(row).all():
            final_mat[:, i] = np.zeros((num_users,)) + 1000000

    return final_mat


def run_augmented_knn_compress(k1: int, k2: int):
    # loading the test data and validation data
    test_data = load_public_test_csv("./data")
    # val_data = load_valid_csv("./data")
    # metadata = load_meta("./data/question_meta.csv")
    # sparse_matrix = load_train_sparse("./data").toarray()
    # num_users, num_questions = sparse_matrix.shape
    # final_mat = compute_subject_matrix()

    val_data = load_valid_csv("./data")
    metadata = load_meta("./data/question_meta.csv")
    sparse_matrix = load_train_sparse("./data").toarray()
    num_users, num_questions = sparse_matrix.shape
    final_mat = compute_subject_matrix()

    ###
    # Perform KNN on the reduced student-subject
    # matrix to fill in missing values (imputation)
    ###
    nbrs = KNNImputer(n_neighbors=k1)
    fitted_values = nbrs.fit_transform(final_mat)

    ###
    # Fill in missing values in the original sparse
    # matrix by inference of the student/subject matrix,
    # by looking for question and a student how
    # well the student performed on average
    # on the subjects related to that question
    ###
    sparse_check = np.copy(sparse_matrix)
    for i in range(num_users):
        for j in range(num_questions):
            ans = sparse_matrix[i, j]
            if np.isnan(ans):
                subjects = metadata[j]
                sparse_check[i, j] = \
                    sum([fitted_values[i][subj] for subj in subjects]) / len(
                        subjects)

    print(
        f"Validation Accuracy with 1st KNN: {sparse_matrix_evaluate(val_data, sparse_check)}")
    print(
        f"Test Accuracy with 1st KNN: {sparse_matrix_evaluate(test_data, sparse_check)}")

    ### (Optional Step)
    # Fill in missing values in the original sparse
    # matrix by inference already filled sparse matrix
    # by the step above, effectively using that
    # filled matrix as the training data.
    ###
    nbrs = KNNImputer(n_neighbors=k2)
    nbrs.fit(sparse_check)
    fmat = nbrs.transform(sparse_matrix)
    val_acc = sparse_matrix_evaluate(val_data, fmat)
    test_acc = sparse_matrix_evaluate(test_data, fmat)
    print(
        f"Validation Accuracy with 2nd KNN: {val_acc}")
    print(
        f"Test Accuracy with 2nd KNN: {test_acc}")

    return val_acc, test_acc


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 1, figsize=(10, 20), layout='constrained')

    k1_lst = [11, 20, 35, 40]
    k2_lst = [11, 20, 35, 40]

    val_accs_k1_fixed = []
    test_accs_k1_fixed = []

    # Fixed k1 at 35.
    for k2 in k2_lst:
        val_acc, test_acc = run_augmented_knn_compress(35, k2)
        val_accs_k1_fixed.append(val_acc)
        test_accs_k1_fixed.append(test_acc)

    axs[0].plot(k2_lst, val_accs_k1_fixed, label='Validation Accuracy')

    axs[0].plot(k2_lst, test_accs_k1_fixed, label='Test Accuracy')
    axs[0].set_xlabel('k2')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracies at k1 = 35')
    axs[0].legend()

    val_accs_k2_fixed = []
    test_accs_k2_fixed = []

    # Fixed k2 at 35.
    for k1 in k1_lst:
        val_acc, test_acc = run_augmented_knn_compress(k1, 35)
        val_accs_k2_fixed.append(val_acc)
        test_accs_k2_fixed.append(test_acc)

    axs[1].plot(k1_lst, val_accs_k2_fixed, label='Validation Accuracy')

    axs[1].plot(k1_lst, test_accs_k2_fixed, label='Test Accuracy')
    axs[1].set_xlabel('k1')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracies at k2 = 35')
    axs[1].legend()

    plt.show()



