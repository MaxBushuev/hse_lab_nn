import numpy as np
import argparse

from dataset import MnistDataloader, random_split_train_val
from model import TwoLayerNet
from trainer import Trainer, Dataset
from optim import SGD, MomentumSGD
from metrics import multiclass_accuracy


def prepare_for_neural_network(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(float) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(float) / 255.0

    mean_image = np.mean(train_flat, axis = 0)
    train_flat -= mean_image
    test_flat -= mean_image
    
    return train_flat, test_flat


def main(args):
    mnist_loader = MnistDataloader(args.train_images_path,
                                   args.train_labels_path,
                                   args.test_images_path,
                                   args.test_labels_path
    )

    train_X, train_y, test_X, test_y = mnist_loader.load_data()
    train_X, test_X = prepare_for_neural_network(train_X, test_X)
    train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val = 1000)

    model = TwoLayerNet(n_input = train_X.shape[1], n_output = 10, hidden_layer_size = 128, reg = 1e-4)
    dataset = Dataset(train_X, train_y, val_X, val_y)
    trainer = Trainer(model, dataset, MomentumSGD(), learning_rate=1e-3, learning_rate_decay=0.95, num_epochs=10)

    loss_history, train_history, val_history = trainer.fit()

    test_pred = model.predict(test_X)
    test_accuracy = multiclass_accuracy(test_pred, test_y)
    print('Neural net test set accuracy: %f' % (test_accuracy, ))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_images_path", type=str)
    parser.add_argument("--train_labels_path", type=str)
    parser.add_argument("--test_images_path", type=str)
    parser.add_argument("--test_labels_path", type=str)

    args = parser.parse_args()

    main(args)
