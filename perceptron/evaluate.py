import numpy as np

from model.perceptron import SimplePerceptron
from utils import load_data

PARAMS = {
    'init_weights': [-0.1, 0.1],
    'alpha_lr': 0.5
}


def run_training(perceptron, params, x_train, y_train, activation='bipolar'):
    perceptron.init_state(low=init_weights[0], high=init_weights[1])
    epochs = perceptron.train(x_train, y_train, params['alpha_lr'], epochs=1000)
    print("Learning epochs: {}".format(epochs))

    return perceptron.get_state()


def run_eval(perceptron, checkpoint, input_data, activation='bipolar'):
    pass


if __name__ == '__main__':
    params = PARAMS
    unipolar_x, unipolar_y = load_data('data/unipolar_train.json', logic_gate='AND')
    unipolar_ext_x, unipolar_ext_y = load_data('data/unipolar_ext_train.json', logic_gate='AND')
    unipolar_x = np.concatenate((unipolar_x, unipolar_ext_x), axis=0)
    unipolar_y = np.concatenate((unipolar_y, unipolar_ext_y), axis=None)

    bipolar_x, bipolar_y = load_data('data/bipolar_train.json', logic_gate='AND')
    bipolar_ext_x, bipolar_ext_y = load_data('data/bipolar_ext_train.json', logic_gate='AND')
    bipolar_x = np.concatenate((bipolar_x, bipolar_ext_x), axis=0)
    bipolar_y = np.concatenate((bipolar_y, bipolar_ext_y), axis=None)
    init_weights = params['init_weights']

    perceptron = SimplePerceptron(unipolar_x.shape, activation='unipolar')
    perceptron.init_state(low=init_weights[0], high=init_weights[1])
    perceptron.train(unipolar_x, unipolar_y, alpha_lr=params['alpha_lr'], epochs=10000)
    print(perceptron.pred([0.9, 0.9]))
