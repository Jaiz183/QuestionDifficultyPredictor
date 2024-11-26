from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


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

    # Better version! Less convoluted.
    # Ensure that NaN entries are not counted in sum.
    # Theta gradient comp.
    del_theta_one = np.where(np.isnan(data), 0, data) @ np.ones(
        (M, 1))
    del_theta_two = np.where(np.isnan(data), 0, sigmoid(
        params_matrix)) @ np.ones((M, 1))
    del_theta = del_theta_one - del_theta_two

    # Beta gradient comp.
    del_beta_one = np.where(np.isnan(data.T), 0, data.T) @ np.ones(
        (N, 1))
    del_beta_two = np.where(np.isnan(data.T), 0, sigmoid(
        params_matrix).T) @ np.ones((N, 1))
    del_beta = -del_beta_one + del_beta_two

    # Old version which is more convoluted.
    # params_matrix = -params_matrix
    #
    # data_complement = 1 - data
    # # Replace all NaN spots with 0 to with-hold from sum.
    # del_theta_one = np.where(np.isnan(data), 0, data_complement) @ np.ones(
    #     (M, 1))
    # del_theta_two = np.where(np.isnan(data), 0, sigmoid(
    #     params_matrix)) @ np.ones((M, 1))
    # del_theta = -del_theta_one + del_theta_two
    #
    # del_beta_one = np.where(np.isnan(data.T), 0, data_complement.T) @ np.ones(
    #     (N, 1))
    # del_beta_two = np.where(np.isnan(data.T), 0, sigmoid(
    #     params_matrix).T) @ np.ones((N, 1))
    # del_beta = del_beta_one - del_beta_two

    # Grad. desc. update.
    theta = theta + lr * del_theta
    beta = beta + lr * del_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    :param data: a sparse matrix.
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta,
    beta, val_acc_lst,
    negative llds for train dataset, negative llds for valid dataset)
    """
    N = len(data)
    M = len(data[0])
    theta = np.zeros((N, 1))
    beta = np.zeros((M, 1))

    # Convert val_data to matrix.
    val_data_matrix = np.empty((N, M))
    users = val_data['user_id']
    questions = val_data['question_id']
    is_correct = val_data['is_correct']
    for i in range(len(is_correct)):
        val_data_matrix[users[i]][questions[i]] = is_correct[i]

    val_acc_lst = []
    neg_lld_train_lst = []
    neg_lld_valid_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data_matrix, theta=theta,
                                         beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        neg_lld_train_lst.append(neg_lld)
        neg_lld_valid_lst.append(neg_lld_val)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, neg_lld_train_lst, neg_lld_valid_lst


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


def plot_statistics(val_accs, train_llds, val_llds):
    fig, axs = plt.subplots(3, 1, figsize=(10, 20), layout='constrained')
    num_iters = list(range(len(val_accs)))
    axs[0].plot(num_iters, val_accs)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Validation Acc.')
    axs[0].set_title('Validation Accuracy')

    axs[1].plot(num_iters, train_llds)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Negative LLD')
    axs[1].set_title('Training Negative LLD')

    print(val_llds)
    axs[2].plot(num_iters, val_llds)
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Negative LLD')
    axs[2].set_title('Validation Negative LLD')

    plt.show()


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # (c)
    # Tune learning rate and number of iterations. With the implemented
    # code, report the validation and test accuracy.
    #####################################################################
    # Store hyperparams to retrieve max val accuracy later.
    hyperparams = []
    learning_rates = [0.0001]
    # learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    num_iters = [200]
    # num_iters = [25, 50, 100, 200]
    for learning_rate in learning_rates:
        for num_iter in num_iters:
            theta, beta, val_accs, neg_llds_train, neg_llds_val = irt(
                sparse_matrix, val_data,
                lr=learning_rate, iterations=num_iter)
            hyperparams.append(
                (val_accs, theta, beta, neg_llds_train, neg_llds_val))
            print(
                f'LR {learning_rate} and NR {num_iter} has validation accuracy {val_accs[-1]}')

    val_accs, theta, beta, neg_llds_train, neg_llds_val = max(hyperparams,
                                                              key=lambda elt:
                                                              elt[0][-1])
    # print(
    # f'theta = {theta}, beta = {beta} have max validation accuracy {val_accs[-1]}')

    # Plot.
    # print(neg_llds_val)
    plot_statistics(val_accs, neg_llds_train, neg_llds_val)

    #####################################################################
    # (d)
    #####################################################################
    M = len(sparse_matrix[0])
    N = len(sparse_matrix)

    # Pick questions
    questions = np.random.choice(np.arange(M), 3)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
