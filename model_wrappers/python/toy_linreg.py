import numpy as np
import os
import rpc
import sys
from scipy import linalg

class ToyLinregModel(rpc.ModelWrapperBase):
    def __init__(self, W):
        self.W = W

    def predict_floats(self, inputs):
        # inputs = np.array(inputs).reshape(len(inputs), len(inputs[0]))
        # bias = np.ones((len(inputs), 1))
        # inputs_with_bias = np.hstack((bias, inputs))
        # print('using weight: ', self.W)
        # return inputs_with_bias.dot(self.W)
        print('using weight: ', self.W)
        return np.array(inputs).dot(self.W)

    def retrain(self, inputs, outputs, weights):
        # inputs = np.array(inputs).reshape(len(inputs), -1)
        # bias = np.ones((len(inputs), 1))
        D = np.diag(weights)
        # X = np.hstack((bias, inputs))
        X = np.array(inputs)
        y = outputs.reshape((len(outputs)), 1)
        xtx = X.T.dot(D).dot(X)
        xty = X.T.dot(D).dot(y)
        print(inputs[:10])
        print(outputs[:10])
        print(weights[:10])
        print('before retrain', self.W)
        self.W = linalg.solve(xtx, xty)
        print('retrained', self.W)

if __name__ == '__main__':
    os.environ['MODEL_PATH'] = '../../../class_project/old_files/toy_data/data/train.npy'
    model_path = os.environ['MODEL_PATH']
    print(model_path)
    model = ToyLinregModel(np.load(model_path))
    rpc.start(model, '127.0.0.1', 32768)
