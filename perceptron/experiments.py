import numpy as np


def weights_init_range_experiment(perceptron_class, activation, repeat, x_train, y_train):
    weights_ranges = np.concatenate((np.arange(0.0, 0.1, 0.05), np.arange(0.1, 1.0, 0.1)),
                                    axis=None)
    # weights_ranges = np.arange(1.0, 4.0, 0.1)
    alpha_lr = 0.05
    perceptron = perceptron_class(x_train.shape, activation)

    experimental_data = []
    for init_range in weights_ranges:
        epochs_counts = []
        for i in range(repeat):
            perceptron.init_state(low=-init_range, high=init_range)
            epochs_count = perceptron.train(x_train, y_train, alpha_lr, epochs=100000)
            epochs_counts.append(epochs_count)
        experimental_data.append(epochs_counts)
    return weights_ranges, experimental_data


def alpha_lr_experiment(perceptron_class, activation, repeat, x_train, y_train):
    weight_init = [-0.1, 0.1]
    # alpha_lr_values = np.concatenate((np.arange(0.01, 0.1, 0.05), np.arange(0.1, 0.5, 0.1)),
    #                                  axis=None)
    alpha_lr_values = np.arange(0.01, 0.3, 0.005)
    # alpha_lr_values = np.arange(0.3, 0.5, 0.005)  #to show gradient explode in Adaline
    perceptron = perceptron_class(x_train.shape, activation)

    experimental_data = []
    for alpha_lr in alpha_lr_values:
        epochs_counts = []
        for i in range(repeat):
            perceptron.init_state(low=weight_init[0], high=weight_init[1])
            epochs_count = perceptron.train(x_train, y_train, alpha_lr, epochs=100000)
            epochs_counts.append(epochs_count)
        experimental_data.append(epochs_counts)

    return alpha_lr_values, experimental_data


def activation_fun_experiment(activation, repeat, x_train, y_train):
    from model.perceptron import SimplePerceptron

    weight_init = [-0.1, 0.1]
    alpha_lr_values = np.arange(0.01, 1.0, 0.05)
    perceptron = SimplePerceptron(x_train.shape, activation)

    experimental_data = []
    for alpha_lr in alpha_lr_values:
        epochs_counts = []
        for i in range(repeat):
            perceptron.init_state(low=weight_init[0], high=weight_init[1])
            epochs_count = perceptron.train(x_train, y_train, alpha_lr, epochs=100000)
            epochs_counts.append(epochs_count)
        experimental_data.append(epochs_counts)

    return alpha_lr_values, experimental_data
