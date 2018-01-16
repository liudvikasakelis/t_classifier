import sys
import time

import keras.models

import numpy as np
import pandas as pd


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
batch = 128

model_path = sys.argv[1]
data_path = sys.argv[2]
cutoff = int(sys.argv[3])

print('Loading model')
model = keras.models.load_model(model_path)
print('Model loaded')

print("Loading the data sets...")
start_time = time.time()

fdata = pd.read_csv(data_path, sep=',', quotechar='"')
fdata = pd.DataFrame(fdata[['y', 'paymentpurpose','operationinout',
                            'operationdate']])
fdata.columns = ['y', 'paymentpurpose', 'operationinout', 'operationdate']
fdata['paymentpurpose'] = (fdata['operationinout'].map(str)
                                   + fdata['paymentpurpose'])
X = [matrixer(x) for x in fdata['paymentpurpose']]
X = [X[i] for i, date in enumerate(fdata.operationdate) if date < cutoff]
Y = fdata.y.tolist()
Y = [Y[i] for i, date in enumerate(fdata.operationdate) if date < cutoff]
Y = Y_matrixer(Y)
print("Loadded all data in {}".format(time.time() - start_time))

print('lengths {} {} (for sanity)'.format(len(X), len(Y)))

for i in range(1, len(X)):
    if len(X[i]) > l0:
        X[i] = X[i][:l0]

print(model.metrics_names)
print(model.evaluate(X, Y, batch_size=batch, verbose=True))
