

import sys
import time

global_start_time = time.asctime()

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D, MaxPooling1D, Embedding, Dropout
from keras.layers import ThresholdedReLU
from keras.optimizers import Adam

from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import json

### Functions

def matrixer(line):
    try:
        line = line.lower()
    except AttributeError:
        line = ''
    l = [ord(x) if ord(x) < c['alphabet_size'] else 20 for x in line]
    l = l + [0] * (c['l0'] - len(l))
    return(l)

def Y_matrixer(Y):
    ret = np.zeros([len(Y), c['num_of_classes']])
    for ai, a in enumerate(Y):
        ret[ai, a-1] = 1
    return(ret)

def Y_unmatrixer(Y):
    Y = [max(enumerate(x), key=(lambda x:x[1]))[0]  for x in Y]
    return(Y)

def conf2str(conf):
    conf = conf.astype('str')
    return('\n'.join([','.join(i) for i in conf]) + '\n')
   
def Wget(fname):
    with open(fname, 'r') as f:
        lines = f.read()
    weights = [int(x) for x in lines.split('\n') if x != '']
    return(weights)

def write_results(result):
    with open(results_file, 'a') as f:
        f.write(result)
        f.write('\n')
    
###

train_file = sys.argv[1]
custom_cfg = sys.argv[2]
if len(sys.argv) > 3:
    with open(sys.argv[3], 'r') as f:
        class_weights = [float(x) if x != '' else 1 
                         for x in f.read().split('\n')]

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
              'CV_cutoff': 16801
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
fdata = fdata[fdata.operationdate < c['train_cutoff']]
fold_mask = fdata.fold.tolist()
X = [matrixer(x) for x in fdata['paymentpurpose']]
Y = fdata.y.tolist()
for i in range(1, len(X)):
    if len(X[i]) > c['l0']:
        X[i] = X[i][:c['l0']]

print("Loadded all data in {}".format(time.time() - start_time))
print('lengths {} {}'.format(len(X), len(Y)))

write_results('\n'.join([global_start_time, json.dumps(c)]))
write_results('fold,epoch,categorical_crossentropy,categorical_accuracy')

for current_fold in range(1, 11):
    print('\nDoing fold {}'.format(current_fold))
    start_time = time.time()
    test_index = ((fdata.fold == current_fold) 
                  & (fdata.operationdate > c['CV_cutoff']))
    train_index = ((fdata.fold != current_fold) 
                   & (fdata.operationdate < c['CV_cutoff']))
    X_train = [X[i] for i, value in enumerate(train_index) if value]
    Y_train = [Y[i] for i, value in enumerate(train_index) if value]
    X_test = [X[i] for i, value in enumerate(test_index) if value]
    Y_test = [Y[i] for i, value in enumerate(test_index) if value]
    if len(sys.argv) < 4:
        class_counter = Counter(Y_train)
        class_weights = [pow(len(Y_train)/class_counter[x], 0.5) 
                        if class_counter[x] != 0 else 0 
                        for x in range(1, c['num_of_classes']+1)]

    Y_train = Y_matrixer(Y_train)
    Y_test = Y_matrixer(Y_test)
    
    
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
                  metrics=['categorical_accuracy'], 
                  loss_weights=[class_weights])
    print("New model built")

    for epoch in range(1, c['epochs'] + 1):
        print('Manual epoch {}/{}'.format(epoch, c['epochs']))
        model.fit(X_train, Y_train, epochs=1, 
                  batch_size=c['batch_size'],
                  class_weight=dict((i, a) for i,a in enumerate(class_weights)))
        ev_res = model.evaluate(X_test, Y_test, verbose=0)
        
        pred_Y = Y_unmatrixer(model.predict(X_test))
        conf = confusion_matrix(y_true=Y_unmatrixer(Y_test), y_pred=pred_Y,
                                labels=range(1, c['num_of_classes']+1))
        ev_res.insert(0, epoch)
        ev_res.insert(0, current_fold)
        ev_res[2] = sum(ev_res[2])
        print(ev_res)
        write_results(','.join([str(x) for x in ev_res]))
        write_results(conf2str(conf))
    
    model.save('model{}.test'.format(current_fold))
    print("Done with fold {}!\n".format(current_fold))

write_results(time.asctime())
