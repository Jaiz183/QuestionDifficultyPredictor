"""
Augmentation.
"""
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


def add_metadata() -> np.ndarray:
    """
    Returns data matrix with metadata added.
    """
    ...


def compute_pcs(k) -> np.ndarray:
    """
    Returns first k PCs of data.
    """
    ...


def load_meta(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {"question_id": [], "subject_id": []}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # print(row)
            try:
                data["question_id"].append(int(row[0]))
                subjs = ast.literal_eval(row[1])[1:]
                if 1 in subjs:
                    subjs = subjs[1:]

                data["subject_id"].append(subjs)
                # print(subjs)
            except ValueError:
                print(f"Value Error: {row}")
                pass
            except IndexError:
                print(f"Index Error: {row}")
                pass
    return data


def knn_subset_questions(train_data: np.ndarray):
    """
    ...
    """
    # Ensure that each question is a row.
    train_data = train_data.T
    final_matrix = np.zeros(train_data.shape)
    metadata = load_meta('data/question_meta.csv')
    question_ids = metadata['question_id']
    subject_ids = metadata['subject_id']
    num_questions = len(question_ids)
    NUM_NEIGHBOURS = 11
    num_students = train_data.shape[1]

    for i in range(num_questions):
        curr_subject_id = subject_ids[i]
        curr_question_id = question_ids[i]
        # Get subset of questions that have one subject in common with current question.
        subset = [question_ids[j] for j in range(num_questions) if
                  set(subject_ids[j]) & set(curr_subject_id) and i != j]
        knn = KNNImputer(n_neighbors=NUM_NEIGHBOURS, keep_empty_features=True)

        # First entry should be current question (question to be predicted).
        curr_train_data = np.zeros((len(subset) + 1, num_students))
        curr_train_data[0] = train_data[curr_question_id]
        curr_train_data[1:] = train_data[subset]

        # Get prediction for question id and replace in final preds.
        preds = knn.fit_transform(curr_train_data)
        # print(final_matrix.shape)
        # print(curr_train_data.shape)
        # print(preds.shape)
        final_matrix[curr_question_id] = preds[0]

    return final_matrix


def knn_subset_disjoint_sets(train_data: np.ndarray):
    # Ensure that each question is a row.
    train_data = train_data.T
    final_matrix = np.zeros(train_data.shape)
    metadata = load_meta('data/question_meta.csv')
    question_ids = metadata['question_id']
    subject_ids = metadata['subject_id']
    num_questions = len(question_ids)
    NUM_NEIGHBOURS = 11
    num_students = train_data.shape[1]

    # TODO: remove the first subject.
    disj_sets_subjects = DisjointSet()
    for i in range(num_questions):
        curr_subject_id = subject_ids[i]
        curr_question_id = question_ids[i]

        # Add all subjects.
        last_subject = curr_subject_id[0]
        disj_sets_subjects.add(curr_subject_id[0])
        for j in range(1, len(curr_subject_id)):
            subject = curr_subject_id[j]
            disj_sets_subjects.add(subject)
            disj_sets_subjects.merge(subject, last_subject)
            last_subject = subject

    subsets = disj_sets_subjects.subsets()
    # print(subsets)
    questions = [[] for _ in range(len(subsets))]
    # print(questions)
    for i in range(num_questions):
        curr_subject_id = subject_ids[i]
        curr_question_id = question_ids[i]

        for j in range(len(subsets)):
            # print(set(curr_subject_id) & subsets[j] != set())
            if set(curr_subject_id) & subsets[j]:
                questions[j].append(curr_question_id)

    for question in questions:
        knn = KNNImputer(n_neighbors=11, keep_empty_features=True)
        preds = knn.fit_transform(train_data[question])
        final_matrix[question] = preds

    return final_matrix



    # print(questions)


if __name__ == "__main__":
    train_data = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # preds = knn_subset_questions(train_data)
    # val_accs = sparse_matrix_evaluate(val_data, preds.T)
    # test_accs = sparse_matrix_evaluate(test_data, preds)
    # print(val_accs)
    # print(test_accs)

    preds = knn_subset_disjoint_sets(train_data)
    val_accs = sparse_matrix_evaluate(val_data, preds.T)
    test_accs = sparse_matrix_evaluate(test_data, preds.T)
    print(val_accs)
    print(test_accs)
