from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: a sparse matrix
    :param theta: N by 1 Vector
    :param beta: M by 1 Vector
    :return: float
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    # log_lklihood = 0.0
    N = len(theta)
    M = len(beta)
    # print(beta)
    # print(theta)
    param_matrix = theta @ np.ones((1, M)) - np.ones((N, 1)) @ beta.T
    # print(np.where(np.isnan(data), 0, data * param_matrix))
    first_term = np.sum(
        np.where(np.isnan(data), 0, param_matrix * data)
    )
    # print(first_term)
    second_term = np.sum(
        np.where(np.isnan(data), 0, np.logaddexp(0, param_matrix)))
    log_lklihood = first_term - second_term
    # print(log_lklihood)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: a sparse matrix
    :param lr: float learning rate
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # Implement the function as described in the docstring.             #
    #####################################################################
    N = len(theta)
    M = len(beta)
    params_matrix = theta @ np.ones((1, M)) - np.ones((N, 1)) @ beta.T

    data_complement = 1 - data
    # Replace all NaN spots with 0 to with-hold from sum.
    del_theta_one = np.where(np.isnan(data), 0, data_complement) @ np.ones(
        (M, 1))
    del_theta_two = np.where(np.isnan(data), 0, sigmoid(
        params_matrix)) @ np.ones((M, 1))
    del_theta = -del_theta_one + del_theta_two

    del_beta_one = np.where(np.isnan(data.T), 0, data_complement.T) @ np.ones(
        (N, 1))
    del_beta_two = np.where(np.isnan(data.T), 0, sigmoid(
        params_matrix).T) @ np.ones((N, 1))
    del_beta = del_beta_one - del_beta_two

    theta = theta + lr * del_theta
    beta = beta + lr * del_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: a sparse matrix.
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    N = len(data)
    M = len(data[0])
    theta = np.zeros((N, 1))
    beta = np.zeros((M, 1))

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(
        data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # learning_rates = [0.05]
    learning_rates = [0.05, 0.1, 0.2]
    # num_iters = [100]
    num_iters = [50, 100, 200]
    for learning_rate in learning_rates:
        for num_iter in num_iters:
            theta, beta, val_accs = irt(sparse_matrix, val_data,
                                        lr=learning_rate, iterations=num_iter)
            print(
                f'LR {learning_rate} and NR {num_iter} has validation accuracy {val_accs[-1]}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
