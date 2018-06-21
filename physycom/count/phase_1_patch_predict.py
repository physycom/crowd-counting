# predict patch image's count using trained fully connected model
import numpy as np
import scipy.io as sci
import sys
from keras.models import model_from_json

if len(sys.argv) != 3:
	print("Usage : %s path/to/phase_0/mat path/to/model/basename")
	exit(1)

inputmat = sys.argv[1]
test_data = sci.loadmat(inputmat)
n = int(test_data['features'].size/1000)
X_test = test_data['features'].reshape(n, 1000)
print("Data loaded...")

# load model
modelbasename = sys.argv[2]
with open(modelbasename + ".json", 'r') as mod:
	modj = mod.read()
model = model_from_json(modj)

# load weights
model.load_weights(modelbasename + ".h5")
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
print("Model loaded...")

# prediction
predictions = model.predict(X_test, batch_size=1000, verbose=0)
sci.savemat(inputmat.split(".")[0] + ".phase_1.mat", {'predictions':predictions})
