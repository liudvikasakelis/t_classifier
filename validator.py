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
batch = 512

model_path = sys.argv[1]
data_path = sys.argv[2]
# cutoff1 = int(sys.argv[3])
# cutoff2 = int(sys.argv[4])

print('Loading {}'.format(model_path))
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

for i in range(1, len(X)):
    if len(X[i]) > l0:
        X[i] = X[i][:l0]
Y = fdata.y.tolist()

print("Loadded all data ({} lines) in {}".format(len(X), 
                                                 time.time() - start_time))

for cutoff1, cutoff2 in [(12000, 16100), (16100, 16750), (16750, 17500)]:
    X_t = [X[i] for i, date in enumerate(fdata.operationdate) if cutoff1 <= date < cutoff2]
    Y_t = [Y[i] for i, date in enumerate(fdata.operationdate) if cutoff1 <= date < cutoff2]
    Y_t = Y_matrixer(Y_t)
    print('date range {} - {}, total entries {}'.format(cutoff1, cutoff2, 
                                                        len(X_t))) 
    print(model.metrics_names)
    print(model.evaluate(X_t, Y_t, batch_size=batch, verbose=True))
