import numpy as np


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.sum(prediction == ground_truth) / len(prediction)

    return accuracy
