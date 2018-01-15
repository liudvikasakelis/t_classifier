import sys
import time

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.optimizers import Adam
import keras

import numpy as np
import pandas as pd


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

def matrixer(line):
    try:
        line = line.lower()
    except AttributeError:
        line = ''
    l = [ord(x) if ord(x) < alphabet_size else 20 for x in line]
    l = l + [0] * (l0 - len(l))
    return(l)

def Y_matrixer(Y):
    ret = np.zeros([len(Y), 112])
    for ai, a in enumerate(Y):
        ret[ai, a-1] = 1
    return(ret)

alphabet_size = 400
l0 = 125
    
model_path = sys.argv[1]
data_path = sys.argv[2]
print('Loading model')
model = keras.models.load_model(model_path)
print('Model loaded')

print("Loading the data sets...")
start_time = time.time()

full_train_data = pd.read_csv(data_path, sep=',', quotechar='"')
full_train_data = pd.DataFrame(full_train_data[['y', 'paymentpurpose',
                                                'operationinout', 'operation_date_less_than_2016_01_01']])
full_train_data.columns = ['y', 'paymentpurpose', 'operationinout', 'old']
full_train_data['paymentpurpose'] = (full_train_data['operationinout'].map(str)
                                   + full_train_data['paymentpurpose'])
                                   
X = [matrixer(x) for x in full_train_data['paymentpurpose']]
Y = full_train_data.y.tolist()
Y = Y_matrixer(Y)
print("Loadded all data in {}".format(time.time() - start_time))

print('lengths {} {} (for sanity)'.format(len(X), len(Y)))
#print(X[1])
#print(Y[1:100])
for i in range(1, len(X)):
    if len(X[i]) > l0:
        X[i] = X[i][:l0]

print(model.metrics_names)
print(model.evaluate(X, Y, batch_size=1024))