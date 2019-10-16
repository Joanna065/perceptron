import json
import os

import matplotlib.pyplot as plt
import numpy as np

PATH = '/home/joanna/Desktop/SIECI NEURONOWE/Sieci neuronowe/Laboratorium/SPRAWOZDANIA/ćwiczenie_1'


def ensure_dir_path_exists(path):
    """Checks if path is an existing directory and if not, creates it."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_data(json_file, logic_gate='AND'):
    """
    Loads train data saved in json files
    :param json_file: path to json
    :param logic_gate: name of logic gate to train perceptron
    :return: tuple (x_train, y_train)
    """
    assert logic_gate in ['AND', 'OR']
    with open(json_file, "r") as f:
        data = json.load(f)
        gate_data = data[logic_gate]
    return np.asarray(gate_data['x'], dtype=np.float16), np.asarray(gate_data['y'],
                                                                    dtype=np.float16)


def plot_data_with_std(x, y, x_label, y_label, filename=None, dirname=None, color='green'):
    plt.tight_layout()
    plt.figure(figsize=(10, 5))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    y_means = np.mean(y, axis=1)
    y_std = np.std(y, axis=1)
    plt.errorbar(x, y_means, yerr=y_std, linestyle='None', capsize=3, color=color, fmt='-o')

    if filename is not None:
        if dirname is not None:
            ensure_dir_path_exists(os.path.join(PATH, dirname))
            plt.savefig(os.path.join(PATH, dirname, filename) + '.png')
        else:
            plt.savefig(os.path.join(PATH, filename) + '.png')
    plt.show()


def plot_compare_activation_fun(x, y1, y2, x_label, y_label, label_1, label_2, filename, dirname,
                                color_1='green', color_2='blue'):
    plt.tight_layout()
    plt.figure(figsize=(10, 5))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    y1_means = np.mean(y1, axis=1)
    y1_std = np.std(y1, axis=1)
    plt.errorbar(x, y1_means, yerr=y1_std, linestyle='None', capsize=3, color=color_1, fmt='-o',
                 label=label_1)

    y2_means = np.mean(y2, axis=1)
    y2_std = np.std(y2, axis=1)
    plt.errorbar(x, y2_means, yerr=y2_std, linestyle='None', capsize=3, color=color_2, fmt='-o',
                 label=label_2)

    plt.legend()

    if filename is not None:
        if dirname is not None:
            ensure_dir_path_exists(os.path.join(PATH, dirname))
            plt.savefig(os.path.join(PATH, dirname, filename) + '.png')
        else:
            plt.savefig(os.path.join(PATH, filename) + '.png')
    plt.show()
