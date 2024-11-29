import os
import csv
import ast
from pprint import pprint 
from utils import _load_csv, load_train_sparse, \
    load_valid_csv, sparse_matrix_evaluate, load_public_test_csv
import numpy as np 
from sklearn.impute import KNNImputer
from ensemble import bagging 

# manually seen form csv
NUM_SUBJ = 388

test_data = load_public_test_csv("./data")
def load_meta(path):
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
                print(f"Value Error: {row}")
                pass
            except IndexError:
                print(f"Index Error: {row}")
                pass
    return data


val_data = load_valid_csv("./data")

def compute_subject_matrix():
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
                new_mat[user_id][sub_id] += answer
                subj_count[user_id, sub_id] += 1
            
    subj_count[subj_count == 0] = np.nan

    contributions = np.zeros(sparse_matrix.shape)
    for i in range(num_users):
        for j in range(num_questions):
            ans = sparse_matrix[i,j]
            if np.isnan(ans):
                continue
            contributions[i][j] += ans*len(metadata[j])
    assert np.nansum(new_mat) == np.nansum(contributions)

    final_mat = np.divide(new_mat, subj_count)

    for i in range(NUM_SUBJ):
        row = final_mat[:,i]
        if np.isnan(row).all():
            final_mat[:,i] = np.zeros((num_users,)) + 100000000000
            # print("MAKARENA")

    return final_mat


val_data = load_valid_csv("./data")

metadata = load_meta("./data/question_meta.csv")
sparse_matrix = load_train_sparse("./data").toarray()
num_users, num_questions = sparse_matrix.shape
final_mat = compute_subject_matrix()

k = 11
nbrs = KNNImputer(n_neighbors=k)
fitted_values = nbrs.fit_transform(final_mat)

# 

sparse_check = np.copy(sparse_matrix)

for i in range(num_users):
    for j in range(num_questions):
        ans = sparse_matrix[i,j]
        if np.isnan(ans):
            subjects = metadata[j]
            sparse_check[i,j] = \
                sum([fitted_values[i][subj] for subj in subjects])/len(subjects)

print(sparse_matrix_evaluate(val_data, sparse_check))


bags = bagging(sparse_check, 5)

mmm = np.zeros(sparse_matrix.shape)
for bag in bags:
    nbrs = KNNImputer(n_neighbors=k)
    nbrs.fit(bag)
    fmat = nbrs.transform(sparse_matrix)
    mmm += fmat

print(sparse_matrix_evaluate(val_data, mmm/5))

nbrs = KNNImputer(n_neighbors=k)
nbrs.fit(sparse_check)
fmat = nbrs.transform(sparse_matrix)
print(sparse_matrix_evaluate(val_data, fmat))






































# kkkk = np.zeros((sparse_matrix.shape))
# c = 0
# for user_id in range(len(sparse_matrix)):
#     user = np.array([sparse_matrix[user_id]])
#     nbrs = KNNImputer(n_neighbors=k)
#     rem = np.random.uniform(low=0, high=500, size=100).astype(int)
#     np.append(rem, user_id)
#     nbrs.fit(np.delete(sparse_check, rem, 0))
#     fmat = nbrs.transform(user)
    
#     kkkk[user_id] = fmat
#     c += 1
#     if c % 10 == 0:
#         print(f"{user_id}---{sparse_matrix_evaluate(val_data, kkkk)}")
# # benchmark = 0.6895




# nbrs = KNNImputer(n_neighbors=k)
# nbrs.fit(sparse_check)
# fmat = nbrs.transform(sparse_matrix)
# print(sparse_matrix_evaluate(val_data, fmat))






# sparse_check = np.copy(sparse_matrix)

# for i in range(num_users):
#     for j in range(num_questions):
#         ans = sparse_matrix[i,j]
#         if np.isnan(ans):
#             subjects = metadata[j]
#             sparse_check[i,j] = \
#                 sum([fitted_values[i][subj] for subj in subjects])/len(subjects)

# print(sparse_matrix_evaluate(val_data, sparse_check))


# bags = bagging(sparse_check, 5)

# nbrs = KNNImputer(n_neighbors=k)
# nbrs.fit(sparse_check)
# fmat = nbrs.transform(sparse_matrix)
# print(sparse_matrix_evaluate(val_data, fmat))
