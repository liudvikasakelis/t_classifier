import sys
import time

global_start_time = time.asctime()
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.optimizers import Adam

import numpy as np
import pandas as pd

config_file = sys.argv[1]
train_file = sys.argv[2]
results_file = 'results/' + config_file.split('/')[-1] + str(int(time.time()/60)) + '.results.txt'
print('Using config file {}, results output to {}'.format(config_file, results_file))

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


configuration = open(config_file).read()
exec(configuration)

conv_layers = config.model.conv_layers
fully_layers = config.model.fully_connected_layers

l0 = config.l0
alphabet_size = config.alphabet_size
num_of_classes = config.num_of_classes
th = config.model.th

embedding_size = config.model.embedding_size

p = config.dropout_p
cutoff = config.date_cutoff

epsilon = config.training.epsilon
alpha = config.training.alpha
beta1 = config.training.beta1
beta2 = config.training.beta2
decay = config.training.decay

# Input layer
inputs = Input(shape=(l0,), name='sent_input', dtype='int64')

# Embedding layer

x = Embedding(alphabet_size + 1, embedding_size, input_length=l0)(inputs)

# Convolution layers
for cl in conv_layers:
    x = Convolution1D(cl[0], cl[1])(x)
    x = ThresholdedReLU(th)(x)
    if not cl[2] is None:
        x = MaxPooling1D(cl[2])(x)

x = Flatten()(x)


#Fully connected layers

for fl in fully_layers:
    x = Dense(fl)(x)
    x = ThresholdedReLU(th)(x)
    x = Dropout(0.5)(x)

predictions = Dense(num_of_classes, activation='softmax')(x)
model = Model(input=inputs, output=predictions)
optimizer = Adam(lr=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon,
                 decay=decay)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
print("New model built")


print("Loading the data sets...")
start_time = time.time()

# from data_utils import Data

fdata = pd.read_csv(train_file, sep=',', quotechar='"')

fdata = pd.DataFrame(fdata[['fold', 'y', 'paymentpurpose',
                                        'operationinout', 'operationdate']])
fdata.columns = ['fold', 'y', 'paymentpurpose', 'operationinout', 
                           'operationdate']

fdata['paymentpurpose'] = (fdata['operationinout'].map(str)
                                   + fdata['paymentpurpose'])
fold_mask = fdata.fold.tolist()
X = [matrixer(x) for x in fdata['paymentpurpose']]
Y = fdata.y.tolist()

for i in range(1, len(X)):
    if len(X[i]) > l0:
        X[i] = X[i][:l0]

print("Loadded all data in {}".format(time.time() - start_time))
print('lengths {} {} (for sanity)'.format(len(X), len(Y)))

total_eval_results = [['fold', 'epoch', 'categorical_crossentropy',
                       'categorical_accuracy']]

for current_fold in range(1, 11):
    print('\nDoing fold {}'.format(current_fold))
    start_time = time.time()
    test_index = ((fdata.fold == current_fold) & (fdata.operationdate > cutoff))
    train_index = ((fdata.fold != current_fold) & (fdata.operationdate < cutoff))
    print(test_index[1:20])
    print(train_index[1:20])
    X_train = [X[i] for i, value in enumerate(train_index) if value]
    Y_train = [Y[i] for i, value in enumerate(train_index) if value]
    X_test = [X[i] for i, value in enumerate(test_index) if value]
    Y_test = [Y[i] for i, value in enumerate(test_index) if value]

    Y_train = Y_matrixer(Y_train)
    Y_test = Y_matrixer(Y_test)
    print('Folding done in {} \n {} train entries\n {} test entries'.format(
           time.time() - start_time, len(X_train), len(X_test)))
    print('ratio {}'.format(len(X_train)/len(X_test)))
   
    inputs = Input(shape=(l0,), name='sent_input', dtype='int64')
    x = Embedding(alphabet_size + 1, embedding_size, input_length=l0)(inputs)
    for cl in conv_layers:
        x = Convolution1D(cl[0], cl[1])(x)
        x = ThresholdedReLU(th)(x)
        if not cl[2] is None:
            x = MaxPooling1D(cl[2])(x)
    x = Flatten()(x)
    for fl in fully_layers:
        x = Dense(fl)(x)
        x = ThresholdedReLU(th)(x)
        x = Dropout(0.5)(x)
    predictions = Dense(num_of_classes, activation='softmax')(x)
    model = None
    model = Model(input=inputs, output=predictions)
    optimizer = Adam(lr=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon,
                 decay=decay)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
    print("New model built")
    for epoch in range(1, config.training.epochs + 1):
        print('Manual epoch {}/{}'.format(epoch, config.training.epochs))
        model.fit(X_train, Y_train, epochs=1, 
                  batch_size=config.batch_size)
        ev_res = model.evaluate(X_test, Y_test, verbose=0)
        ev_res.insert(0, epoch)
        ev_res.insert(0, current_fold) 
        print(ev_res)
        total_eval_results.append(ev_res)
    
    # model_name = 'model{}.test'.format(current_fold)
    # model.save(model_name)
    print("Done with fold {}!\n".format(current_fold))

total_eval_results = [','.join([str(a) for a in x]) for x in total_eval_results]
with open(results_file, mode='w') as f:
    f.write(global_start_time)
    f.write('\n')
    f.write(time.asctime())
    f.write('\n')
    f.write(configuration)
    f.write('\n'.join([str(a) for a in total_eval_results]))
