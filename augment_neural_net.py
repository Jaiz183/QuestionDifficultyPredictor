import os
import csv
import ast
from pprint import pprint 
from utils import _load_csv, load_train_sparse
import numpy as np 

NUM_SUBJ = 388

def load_meta(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data -- key-value parts of the form qid:subjects
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # print(row)
            try:
                data[int(row[0])] = ast.literal_eval(row[1])
                # data[int(row[0])] = ast.literal_eval(row[1])[1:] # avoiding group 0
            except ValueError:
                print(f"Value Error: {row}")
                pass
            except IndexError:
                print(f"Index Error: {row}")
                pass
    return data


data = _load_csv("./data/train_data.csv")
metadata = load_meta("./data/question_meta.csv")

sparse_matrix = load_train_sparse("./data").toarray()
num_users, num_questions = sparse_matrix.shape

subj_count = np.zeros((NUM_SUBJ,))
new_mat = np.zeros((num_users, NUM_SUBJ))


for user_id in range(num_users):
    for question_id in range(num_questions):
        answer = sparse_matrix[user_id, question_id]
        if np.isnan(answer):
            continue
        for sub_id in metadata[question_id]:
            new_mat[user_id][sub_id] += 1
            subj_count[sub_id] += 1

print(new_mat)

a,b = np.unique(new_mat, return_counts=True)
a,b = np.unique(sparse_matrix, return_counts=True)
print(a,b)