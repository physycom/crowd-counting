# predict patch image's count using trained fully connected model
import numpy as np
import scipy.io as sci
import sys
from keras.models import model_from_json

if len(sys.argv) < 2:
	print("Usage : %s path/to/phase_0/mat")
	exit(1)

inputmat = sys.argv[1]
test_data = sci.loadmat(inputmat)
n = int(test_data['features'].size/1000)
X_test = test_data['features'].reshape(n, 1000)
print("Data loaded...")

# load trained model from disk
json_file = open('../model/model_B_SHT.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("../model/model_B_SHT.h5")
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
print("Model loaded...")

predictions = model.predict(X_test, batch_size=1000, verbose=0)
output = inputmat.split(".")[0] + ".phase_1.mat"
sci.savemat(output, {'predictions':predictions})
