

import sys
import time

global_start_time = time.asctime()

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D, MaxPooling1D, Embedding, Dropout
from keras.layers import ThresholdedReLU
from keras.optimizers import Adam
import keras.backend as K

import collections
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import json

### Functions

def matrixer(line):
    l = [ord(x) if ord(x) < c['alphabet_size'] else 20 for x in line]
    l = l + [0] * (c['l0'] - len(l))
    l = l[:c['l0']]
    return(l)

def Y_matrixer(Y):
    ret = np.zeros([len(Y), c['num_of_classes']])
    for ai, a in enumerate(Y):
        ret[ai, a] = 1
    return(ret)

def Y_unmatrixer(Y):
    Y = [max(enumerate(x), key=(lambda x:x[1]))[0] for x in Y]
    return(Y)

def conf2str(conf):
    conf = conf.astype('str')
    return('\n'.join([','.join(i) for i in conf]) + '\n')
   
def write_results(result):
    with open(results_file, 'a') as f:
        f.write(result)
        f.write('\n')
   
def create_map(y_values):
    y_map = dict((y, i+1) for i, y in enumerate(set(y_values)))
    return(y_map)
    
def calculate_weights(y):
    class_counter = collections.Counter(y)
    tots = len(y)
    class_weights = dict((x, pow(tots/class_counter[x], c['w_factor'])) 
                         if class_counter[x] != 0 else (x, 1) 
                         for x in range(1, c['num_of_classes']+1))
    return(class_weights)

def lr_scaler(x):
    return(pow(x, 0.8))

###

train_file = sys.argv[1]
custom_cfg = sys.argv[2]

default_cfg = {'alpha': 2e-3,
              'decay': 0,
              'beta1': 0.9,
              'beta2': 0.999, 
              'epochs': 6,
              'epsilon': 1e-8,
              'conv_layers': [[256, 7, 3],
                              [256, 7, 3],
                              [256, 3, None],
                              [256, 3, None],
                              [256, 3, None],
                              [256, 3, 3]],
              'fully_connected_layers': [1024, 1024],
              'th': 1e-6,
              'embedding_size': 128,
              'alphabet_size': 400,
              'l0': 125,
              'batch_size': 512,
              'num_of_classes': 112,
              'dropout_p': 0.5,
              'train_cutoff': 17166,
              'CV_cutoff': 16801,
              'w_factor': 0.5
}

custom_cfg = json.loads(custom_cfg)
if (custom_cfg.keys() - default_cfg.keys()):
    sys.exit('wrong key in config')

results_file = 'results/' + str(int(time.time()/60)) + '.results.txt'
print('Custom config {}, results output to {}'.format(custom_cfg, results_file))

# Combine configurations preferring custom_cfg
c = {**default_cfg, **custom_cfg} 
print(c)
    
print("Loading the data sets...")
start_time = time.time()

fdata = pd.read_csv(train_file, sep=',', quotechar='"', 
                    usecols=['fold', 'y', 'paymentpurpose', 
                             'operationdate'])

### Map categories to indices
category_map = create_map(fdata.y)
fdata.y = fdata.y.apply(lambda x: category_map[x])

fdata = fdata[fdata.operationdate < c['train_cutoff']]

fold_mask = fdata.fold.tolist()
X = fdata.paymentpurpose.tolist()
Y = fdata.y.tolist()

print("Loadded all data in {}".format(time.time() - start_time))
print('lengths {} {}'.format(len(X), len(Y)))

write_results('\n'.join([global_start_time, json.dumps(c)]))
write_results('fold,epoch,categorical_crossentropy,categorical_accuracy')


for current_fold in range(1, 11):
    print('\nDoing fold {}'.format(current_fold))
    start_time = time.time()
    
    test = fdata[(fdata.fold == current_fold) &
                 (fdata.operationdate > c['CV_cutoff'])]
    train = fdata[(fdata.fold != current_fold) &
                 (fdata.operationdate < c['CV_cutoff'])]
    train = train.drop_duplicates(['paymentpurpose', 'y'])
    
    X_train = [matrixer(x) for x in train.paymentpurpose]
    X_test = [matrixer(x) for x in test.paymentpurpose]
    Y_train = Y_matrixer(train.y)
    Y_test = Y_matrixer(test.y)
    
    class_weights = calculate_weights(train.y)
    
    print('Folding done in {} \n {} train entries\n {} test entries'.format(
           time.time() - start_time, len(X_train), len(X_test)))
    print('ratio {}'.format(len(X_train)/len(X_test)))
   
    # Input layer
    inputs = Input(shape=(c['l0'],), name='sent_input', dtype='int64')
    # Embedding layer
    x = Embedding(c['alphabet_size'] + 1, c['embedding_size'], 
                                          input_length=c['l0'])(inputs)
    # Convolution layers
    for cl in c['conv_layers']:
        x = Convolution1D(cl[0], cl[1])(x)
        x = ThresholdedReLU(c['th'])(x)
        if not cl[2] is None:
            x = MaxPooling1D(cl[2])(x)
    x = Flatten()(x)
    #Fully connected layers
    for fl in c['fully_connected_layers']:
        x = Dense(fl)(x)
        x = ThresholdedReLU(c['th'])(x)
        x = Dropout(0.5)(x)
    predictions = Dense(c['num_of_classes'], activation='softmax')(x)
    model = Model(input=inputs, output=predictions)
    optimizer = Adam(lr=c['alpha'], beta_1=c['beta1'], beta_2=c['beta2'], 
                     epsilon=c['epsilon'], decay=c['decay'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    print("New model built")
    
    get_a_taste = int(pow(len(X_train), 0.5) * 10)
    h = model.fit(X_train[:get_a_taste], Y_train[:get_a_taste],
                  batch_size=c['batch_size'],
                  class_weight=class_weights,
                  shuffle=True)
    alpha_ratio = c['alpha'] / lr_scaler(h.history['loss'][-1])
    h = None
    
    for epoch in range(1, c['epochs'] + 1):
        if h:
            K.set_value(model.optimizer.lr, 
                        alpha_ratio * lr_scaler(h.history['loss'][-1]))
        print('Manual epoch {}/{}, learning rate {}'.format(
            epoch, c['epochs'], K.eval(model.optimizer.lr)))
        h = model.fit(X_train, Y_train, epochs=1, 
                  batch_size=c['batch_size'],
                  class_weight=class_weights,
                  shuffle=True)
        ev_res = model.evaluate(X_test, Y_test, verbose=0)
        
        pred_Y = Y_unmatrixer(model.predict(X_test))
        conf = confusion_matrix(y_true=Y_unmatrixer(Y_test), y_pred=pred_Y,
                                labels=range(1, c['num_of_classes']+1))
        ev_res.insert(0, epoch)
        ev_res.insert(0, current_fold)
        print(ev_res)
        write_results(','.join([str(x) for x in ev_res]))
        write_results(conf2str(conf))
    
    # model.save('model{}.test'.format(current_fold))
    print("Done with fold {}!\n".format(current_fold))

write_results(time.asctime())
