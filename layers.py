import numpy as np


def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(np.power(W, 2))
    grad = reg_strength * 2 * W

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    if preds.ndim == 1:
        pred = preds - np.max(preds)
    else:
        pred = preds - np.max(preds, axis=1)[:, np.newaxis]

    predictions_cur = np.exp(pred)
    predictions_sum = np.sum(np.exp(pred), axis=preds.ndim - 1)

    if preds.ndim == 1:
        probs = predictions_cur / predictions_sum
    else:
        probs = predictions_cur / predictions_sum[:, np.newaxis]

    p = np.zeros(shape= preds.shape)

    if probs.ndim == 1:
        p[target_index] = 1
    else:
        if target_index.ndim == 1:
            p[np.arange(len(p)), target_index] = 1
        else:
            p[np.arange(len(p)), target_index[:, 0]] = 1

    loss = -np.sum(p * np.log(probs))
    d_preds = probs - p

    return loss, d_preds


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        result = np.copy(X)
        result[result < 0] = 0

        return result

    def backward(self, d_out):
        dX = np.zeros(self.X.shape)
        dX[self.X > 0] = 1
        d_result = d_out * dX

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0)[:, np.newaxis].T

        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
