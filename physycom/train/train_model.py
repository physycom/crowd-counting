from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation,LSTM
from keras.optimizers import SGD
import numpy as np
import scipy.io as sci
import numpy as np

# convert features and counts matrix to the format of X, Y
# input
#   features, patch images' feature extracted from ResNet;
#   counts, patch images' count;
# output
#   X, the input of the fully connected regress network with the dimension of M x 1000, which
#   M is the number of the training image patches in formula (1);
#   Y, the output of the fully connected regress network with the dimension of M x 1;
def features2XY(features, counts):
  n = 0
  for c in counts:
    n = n + c.size

  X = np.zeros((n, 1000))
  Y = np.zeros((n, 1))
  k = 0
  for (patch_feature, patch_count) in zip(features, counts):
    X[k:k + patch_count.size, :] = patch_feature.reshape(patch_count.size, 1000)
    Y[k:k + patch_count.size] = patch_count.reshape(patch_count.size, 1)
    k = k + patch_count.size

  return X, Y

#%%
data_mat = '/mnt/e/Alessandro/Peoplebox/new/peoplebox_crowd_dataset/dataset_tiny.mat'
#data_mat = r'E:\Alessandro\Peoplebox\new\peoplebox_crowd_dataset\dataset_tiny.mat'
data = sci.loadmat(data_mat)

# for debugging only reduce dataset size
subsample = 400 # int(0.1*len(data['counts'].shape[1]))
data['counts'] = data['counts'][:,:subsample]
data['features'] = data['features'][:,:subsample]

features = data['features'][0]
counts = data['counts'][0]

# split train and test data
n_test = len(features)
partition = np.arange(n_test)
np.random.seed(44)
np.random.shuffle(partition)
k_test = int(0.1*n_test)
validation_idx = partition[:k_test]
train_idx = partition[k_test:]

# partition data may be needed elsewhere
#sci.savemat('data/partition_UCF.mat', {'partition':partition})
#partition = sci.loadmat('data/partition_UCF.mat')['partition']

# set up train/test arrays
X_train, Y_train = features2XY(features[train_idx], counts[train_idx])
X_test, Y_test = features2XY(features[validation_idx], counts[validation_idx])

# select a model
model_tag="5strati"
if model_tag == "5strati":
  model = Sequential()
  model.add(Dense(100, input_dim=1000))
  model.add(Dense(100))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='relu'))
  model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
elif model_tag == "3strati":
  model = Sequential()
  model.add(Dense(100, input_dim=1000, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='relu'))
  model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

print(model.summary())

# train model
model.fit(X_train, Y_train, epochs=15, batch_size=1000, validation_split=0.25, verbose=0)
result = model.evaluate(X_train, Y_train, batch_size=200, verbose=1, sample_weight=None)
print(result)
result = model.evaluate(X_test, Y_test, batch_size=200, verbose=1, sample_weight=None)
print(result)

# dump model and weights
model_basename = '/mnt/e/Alessandro/Codice/crowd-counting/physycom/model/model_' + model_tag
#model_basename = r'E:\Alessandro\Codice\crowd-counting\physycom\train\model_' + model_tag
model_json = model.to_json()
with open(model_basename + ".json", "w") as json_file:
  json_file.write(model_json)
model.save_weights(model_basename + ".h5")
