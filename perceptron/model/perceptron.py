import numpy as np


class Perceptron(object):
    """
    Base class for perceptron
    """

    def __init__(self, input_shape, activation, use_bias=True, theta=0):
        self._input_shape = input_shape  # N x 2
        self._activation = activation
        self._use_bias = use_bias  # bias (1 * w0)
        self._theta = theta

    def init_state(self, low, high):
        """
        Init trainable variables
        """
        self._W = np.random.uniform(low, high, size=self._input_shape[1:])  # (2,)
        if self._use_bias:
            self._bias = 1

    def get_state(self):
        """
        Return state of trainable variables
        :return: dict of vars
        """
        return dict(W=self._W, bias=self._bias) if self._use_bias else dict(W=self._W)

    def restore_state(self, vars):
        """
        Load previously saved trainable variables
        :param vars: dict of variables
        """
        assert self._W.shape == vars["W"].shape
        self._W = vars["W"]
        if self._use_bias:
            assert self._bias.shape == vars["bias"].shape
            self._bias = vars["bias"]

    def _activation_fun(self, out):
        if self._activation == 'bipolar':
            return np.where(out > self._theta, 1, -1)
        elif self._activation == 'unipolar':
            return np.where(out > self._theta, 1, 0)
        else:
            # or raise an error?
            return out

    def _compute_update(self, x, delta):
        """
        Updates trainable variables during train process
        :param x: input train data
        :param delta: value of update
        """
        self._W = np.add(self._W, self._alpha_lr * x.T @ delta)
        if self._use_bias:
            self._bias = np.add(self._bias, np.sum(self._alpha_lr * delta, axis=0))

    def _net_out(self, x):
        """
        Perceptron prediction for inputs data array
        """
        return np.dot(x, self._W) + self._bias if self._use_bias else np.dot(x, self._W)

    def _stop_condition(self, error):
        raise NotImplementedError

    def train(self, x_train, y_train, alpha_lr, epochs):
        raise NotImplementedError

    def pred(self, x):
        return self._activation_fun(self._net_out(x))


class SimplePerceptron(Perceptron):
    def __init__(self, input_shape, activation='unipolar'):
        super(SimplePerceptron, self).__init__(input_shape=input_shape, activation=activation,
                                               use_bias=True)

    def _stop_condition(self, error):
        return not error.any()

    def train(self, x_train, y_train, alpha_lr, epochs):
        self._alpha_lr = alpha_lr
        epoch_counter = 0

        for epoch in range(epochs):
            y_pred = self._activation_fun(self._net_out(x_train))
            error = y_train - y_pred

            if self._stop_condition(error):
                break
            self._compute_update(x_train, error)
            epoch_counter += 1
        return epoch_counter


class AdalinePerceptron(Perceptron):
    def __init__(self, input_shape, activation='bipolar', error_threshold=0.26):
        self._error_threshold = error_threshold
        super(AdalinePerceptron, self).__init__(input_shape=input_shape, activation=activation,
                                                use_bias=True)

    def _stop_condition(self, error):
        return np.mean(np.power(error, 2)) <= self._error_threshold

    def train(self, x_train, y_train, alpha_lr, epochs):
        self._alpha_lr = alpha_lr
        epoch_counter = 0

        for epoch in range(epochs):
            y_pred = self._net_out(x_train)
            error = y_train - y_pred

            if self._stop_condition(error):
                break
            self._compute_update(x_train, error)
            epoch_counter += 1
        return epoch_counter
