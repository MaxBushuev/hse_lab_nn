import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        self.reg = reg

        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        temp = self.fc1.forward(X)
        temp = self.relu1.forward(temp)
        temp = self.fc2.forward(temp)
        loss, d_preds = softmax_with_cross_entropy(temp, y)

        temp = self.fc2.backward(d_preds)
        dW2 = self.params()['W2'].grad
        dB2 = self.params()['B2'].grad

        temp = self.relu1.backward(temp)
        temp = self.fc1.backward(temp)
        dW1 = self.params()['W1'].grad
        dB1 = self.params()['B1'].grad

        loss_reg1, dW1_reg = l2_regularization(self.params()['W1'].value, self.reg)
        loss_reg2, dW2_reg = l2_regularization(self.params()['W2'].value, self.reg)
        loss_reg3, dB1_reg = l2_regularization(self.params()['B1'].value, self.reg)
        loss_reg4, dB2_reg = l2_regularization(self.params()['B2'].value, self.reg)

        loss += loss_reg1 + loss_reg2 + loss_reg3 + loss_reg4

        self.params()['W1'].grad = dW1 + dW1_reg
        self.params()['W2'].grad = dW2 + dW2_reg
        self.params()['B1'].grad = dB1 + dB1_reg
        self.params()['B2'].grad = dB2 + dB2_reg

        return loss

    def predict(self, X):
        pred = np.zeros(X.shape[0], int)

        pred = self.fc1.forward(X)
        pred = self.relu1.forward(pred)
        pred = self.fc2.forward(pred)

        pred = np.argmax(pred, axis=1)
        
        return pred

    def params(self):
        result = {
            'W1': self.fc1.params()['W'],
            'W2': self.fc2.params()['W'],
            'B1': self.fc1.params()['B'],
            'B2': self.fc2.params()['B'],
        }

        return result
