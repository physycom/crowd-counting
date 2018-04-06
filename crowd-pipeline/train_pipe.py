# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:55:31 2018

@author: nico
"""
# parse command line
import sys
import os
import argparse
# train keras prediction NN with resnet152 features extraction
import numpy as np
from resnet152 import ResNet152
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense

if __name__ == '__main__':
    description = "Crowd Counting Training"
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument("-i", required=False, dest="imagesPath", action="store", help="images path", default="")
    parser.add_argument("-g", required=False, dest="GroundTruthPath", action="store", help="ground truth images path", default="")
    parser.add_argument("-w", required=False, dest="winStep", action="store", help="winStep", default="100")
    parser.add_argument("-x", required=False, dest="xStep", action="store", help="xStep", default="50")
    parser.add_argument("-y", required=False, dest="yStep", action="store", help="yStep", default="50")
    parser.add_argument("-h", required=False, dest="hStep", action="store", help="Height Step", default="50")
    parser.add_argument("-d", required=False, dest="wStep", action="store", help="Width Step", default="50")
    parser.add_argument("-m", required=False, dest="model", action="store", help="Predict model filename (without extension)", default="model/model_venice")
    
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    else:  args = parser.parse_args(sys.argv)

    train_file = [os.path.join(args.imagesPath, f) for f in os.listdir(args.imagesPath) if os.path.isfile(os.path.join(args.imagesPath, f))]
    gt_file    = [os.path.join(args.GroundTruthPath, f) for f in os.listdir(args.GroundTruthPath) if os.path.isfile(os.path.join(args.GroundTruthPath, f))]
    winStep    = int(args.winStep)
    xStep      = int(args.xStep)
    yStep      = int(args.yStep)
    HeightStep = int(args.hStep)
    WidthStep  = int(args.wStep)
    model_out  = args.model

    # Features extraction with resnet152
    features   = np.empty(shape=(len(file), ), dtype=np.object)
    base_model = ResNet152(weights='imagenet', large_input=True)
    model      = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1000').output)
    for (i, f), g in zip(enumerate(train_file), gt_file):
        im = cv2.imread(f)
        height, width, channel = im.shape
        newHeight = int(np.round(height / HeightStep) * 50)
        newWidth  = int(np.round(width / WidthStep) * 50)
        im.resize((newHeight, newWidth, channel))
        if channel == 1:  im = np.merge((im, im, im))
        
        Y = np.arange(0, newHeight, winStep)
        X = np.arange(0, newWidth, winStep)
        features[i] = np.empty(shape=(int(newHeight / HeightStep) - 1, int(newWidth / WidthStep) - 1, 1000), dtype=np.float)
        for row, y in enumerate(Y):
            for column, x in enumerate(X):
                features[i][row, column, :] = model.predict( np.expand_dims( preprocess_input( cv2.resize(
                                                                                                            im[y : y + winStep, x : x + winStep].astype(np.float64), 
                                                                                                            model.input_shape[1:3]) ), 
                                                                            axis=0 ) )
    features = np.concatenate(features, axis=1)


    # define fully connected regress network
    model2 = Sequential()
    model2.add(Dense(100, input_dim=1000))
    model2.add(Dense(100))
    model2.add(Dense(50, activation='relu'))
    model2.add(Dense(50, activation='relu'))
    model2.add(Dense(1, activation='relu'))
    model2.compile(optimizer='Adam',
                   loss='mean_squared_error',
                   metrics=['mean_absolute_error'])
    model2.fit(features, counts, epochs=15, batch_size=1000)
    # Save model 
    with open(model_out + ".json", "w") as json_file: json_file.write(model.to_json())
    model2.save_weights(model_out + ".h5")