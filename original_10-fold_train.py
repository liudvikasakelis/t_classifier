import sys
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.optimizers import Adam
import numpy as np

config_file = sys.argv[1]

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


print("Loading the configurations...")

exec(open(config_file).read())
conv_layers = config.model.conv_layers
fully_layers = config.model.fully_connected_layers
l0 = config.l0
alphabet_size = config.alphabet_size
embedding_size = config.model.embedding_size
num_of_classes = config.num_of_classes
th = config.model.th
p = config.dropout_p
print("Loaded")


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
optimizer = Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
print("New model built")


print("Loading the data sets...")

from data_utils import Data

train_data = Data(data_source = config.train_data_source, 
                     alphabet = config.alphabet,
                     l0 = config.l0,
                     batch_size = 0,
                     no_of_classes = config.num_of_classes)

train_data.loadData()

X_train, y_train = train_data.getAllData()

train_length = X_train.shape[0]

print("Loadded")

slices = 10
step = int(train_length/slices)
start_weights = model.get_weights()

for a in range(0, train_length, step):
    X_val = X_train[a:a+step]
    y_val = y_train[a:a+step]
    subset_X_train = np.concatenate((X_train[:a], X_train[a+step:]), axis=0)
    subset_y_train = np.concatenate((y_train[:a], y_train[a+step:]), axis=0)

    print("\nTraining without {}-{}".format(a, a+step))

    shuffle_weights(model, start_weights)
    print('Weights shuffled')

    model.fit(subset_X_train, subset_y_train, epochs=config.training.epochs, 
              batch_size=config.batch_size) #, validation_data=(X_val, y_val))
    print(model.evaluate(X_val, y_val, verbose=0))
    print("Done!\n")
