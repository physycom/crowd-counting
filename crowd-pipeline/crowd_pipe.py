# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:55:31 2018

@author: nico
"""

# parse command line
import sys
import os
import argparse
# resnet152 keras version and predictNN
import numpy as np
from resnet152 import ResNet152
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model, model_from_json
# IO images and processing
import cv2
# connection with cpp codes
import ctypes

if __name__ == '__main__':
    description = "Crowd Counting Pipeline"
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument("-i", required=False, dest="imagesPath", action="store", help="images path", default="")
    parser.add_argument("-f", required=False, dest="filename", action="store", help="images filename", default="")
    parser.add_argument("-w", required=False, dest="winStep", action="store", help="winStep", default="100")
    parser.add_argument("-x", required=False, dest="xStep", action="store", help="xStep", default="50")
    parser.add_argument("-y", required=False, dest="yStep", action="store", help="yStep", default="50")
    parser.add_argument("-h", required=False, dest="hStep", action="store", help="Height Step", default="50")
    parser.add_argument("-d", required=False, dest="wStep", action="store", help="Width Step", default="50")
    parser.add_argument("-m", required=False, dest="predict_model", action="store", help="Predict Model name (without extension)", default="model_venice")
    parser.add_argument("-p", required=False, dest="MRFParams", action="store", nargs="+", help="MRF Parameters", default=[100., 200., 1.])
    
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    else:  args = parser.parse_args(sys.argv)

    if args.imagesPath == "" and args.filename == "":
        parser.print_help()
        sys.exit(1)
    elif args.imagesPath != "" and args.filename == "":  file = [os.path.join(args.imagesPath, f) for f in os.listdir(args.imagesPath) if os.path.isfile(os.path.join(args.imagesPath, f))]
    elif args.imagesPath == "" and args.filename != "":  file = [args.filename]
    else:  file = iter([args.filename])    
    
    winStep       = int(args.winStep)
    xStep         = int(args.xStep)
    yStep         = int(args.yStep)
    HeightStep    = int(args.hStep)
    WidthStep     = int(args.wStep)
    MRFParams     = ctypes.POINTER(ctypes.c_float * 3)(*list(map(float, args.MRFParams)))
    predict_model = args.predict_model

    ############################################# ONE TIME STEP ################################
    # Import MRF Library
    libmrf  = ctypes.cdll.LoadLibrary("./lib/libmrf.so")
    ptr_im = ctypes.POINTER(ctypes.c_float)
    ptr_int = ctypes.POINTER(ctypes.c_int)
    ptr_flt = ctypes.POINTER(ctypes.c_float * 3)
    libmrf.MRF.argtypes = [ptr_im, ctypes.byref(ctypes.c_int), ctypes.byref(ctypes.c_int), ptr_flt]
    libmrf.MRF.restype  = ptr_int
    # Alternative Evaluate function
    #libmrf.Evaluate.argtypes = [ptr_im, ptr_int, ptr_int, ctypes.byref(ctypes.c_int), ptr_flt]
    #libmrf.Evaluate.restype = ptr_im

    # Import Keras Model for ResNet152
    base_model = ResNet152(weights='imagenet', large_input=True)
    model      = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1000').output)

    # Import and Compile Predict Model extracted from train evaluation
    with open(predict_model + ".json", 'r') as m:  model2 = model_from_json(m.read())
    model2.load_weights(predict_model + ".h5")
    model2.compile( optimizer = "Adam",
                    loss = "mean_squared_error",
                    metrics = ["mean_absolute_error"])

    ############################################################################################

    features   = np.empty(shape=(len(file), ), dtype=np.object)
    finalcount = np.empty(shape=(len(file), ), dtype=np.int)

    # ExtractFeatures with KERAS RESNET152
    for i, f in enumerate(file):
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

    # PATCH PREDICT
    predictions = model2.predict(np.concatenate(features, axis=1), batch_size=1000, verbose=0)

    # EVALUATE
    k = 0
    for i, feature in enumerate(features):
        height, width, _ = feature.shape
        # The marginal data of the predicted count matrix is 0 after apply MRF,
        # so first extending the predicted count matrix by copy marginal data.
        p = predictions[k : k + height * width].reshape(width, height).astype(np.float32).T
        p = cv2.copyMakeBorder( p,
                                top=1, 
                                bottom=1, 
                                left=1, 
                                right=1, 
                                borderType= cv2.BORDER_REPLICATE)
        k += height * width
        # apply MRF
        ############################ TO DO (capture reference and not copy buffer) ##############
        p = np.fromiter(libmrf.MRF(p.ctypes.data_as(ptr_im), ctypes.byref(ctypes.c_int(height)), ctypes.byref(ctypes.c_int(width)), MRFParams) , dtype=np.int32, count=(height+1)*(width+1)).reshape(height+1, width+1)
        p = p[1:height, 1:width]   # remove border
        # FinalCount
        p[range(0, height, 2), range(0, height, 2)] = 0
        if not height % 2: p[height - 1, :] = p[height - 1, :] * .5
        if not width  % 2: p[:, width - 1]  = p[:, width - 1]  * .5
        finalcount[i] = p.sum(dtype=np.int)

    # Alternative Evaluate function
    #heights, widths = zip(*[f.shape[0:2] for f in features])
    #finalcount = np.fromiter(libmrf.Evaluate(np.asarray(predictions).ctypes.data_as(ptr_flt), 
    #                                         np.asarray(heights).ctypes.data_as(ptr_int), 
    #                                         np.asarray(widths).ctypes.data_as(ptr_int),
    #                                         ctypes.byref(ctypes.c_int(len(predictions))),
    #                                         MRFParams),
    #                        dtype=np.float64, count=len(predictions)
    #                        )

    # debug output
    finalcount.savetxt( fname="final_count.csv",
                        delimiter=",",
                        newline="\n",
                        fmt="%d")