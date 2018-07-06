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
# connection with cpp codes
import ctypes
# Image processing matlab-like
from PIL import Image, ImageOps

if __name__ == '__main__':
    description = "Crowd Counting Pipeline"
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument("-i", required=False, dest="imagesPath", action="store", help="images path", default="")
    parser.add_argument("-f", required=False, dest="filename", action="store", help="images filename", default=r"c:/Users/Alessandro/Codice/crowd-counting/new/70_piccoli_omini.jpg")
    parser.add_argument("-w", required=False, dest="winStep", action="store", help="winStep", default="100")
    parser.add_argument("-x", required=False, dest="xStep", action="store", help="xStep", default="50")
    parser.add_argument("-y", required=False, dest="yStep", action="store", help="yStep", default="50")
    parser.add_argument("-e", required=False, dest="hStep", action="store", help="Height Step", default="50")
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

    winStep       = 100#int(args.winStep)
    xStep         = 50#int(args.xStep)
    yStep         = 50#int(args.yStep)
    HeightStep    = 50#int(args.hStep)
    WidthStep     = 50#int(args.wStep)
    MRFParams     = np.asarray([100. , 200., 1.])#np.asarray(list(map(float, args.MRFParams)))
    predict_model = '../model/model_B_SHT'#args.predict_model
    file = ['../new/70_piccoli_omini.JPG'] ################################################ remove me!!!!!!!!!!!!!!!!!!!

    ############################################# ONE TIME STEP ################################
    # Import MRF Library
    libmrf  = ctypes.cdll.LoadLibrary("./lib/mrf.dll")
    ptr_im = ctypes.POINTER(ctypes.c_float)
    ptr_int = ctypes.POINTER(ctypes.c_int)
    ptr_flt = ctypes.POINTER(ctypes.c_float * 3)
    MRF = libmrf.MRF
    MRF.argtypes = [ptr_im, ctypes.c_int, ctypes.c_int, ptr_flt]
    MRF.restype  = ptr_int
    # Alternative Evaluate function
    #libmrf.Evaluate.argtypes = [ptr_im, ptr_int, ptr_int, ctypes.c_int, ptr_flt]
    #libmrf.Evaluate.restype = ptr_im

    # Import Keras Model for ResNet152
    base_model = ResNet152(weights='imagenet', large_input=False)
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
        im = Image.open(f)#im = cv2.imread(f)
        width, height = im.size
        newHeight = int(np.round(height / HeightStep) * 50)
        newWidth  = int(np.round(width / WidthStep) * 50)
        im = im.resize((newWidth, newHeight), Image.BICUBIC | Image.ANTIALIAS) # matlab has default bicubic and anti-aliasing
        #if channel == 1:  im = np.merge((im, im, im)) # not work with pillow

        Y = np.arange(0, newHeight - yStep, yStep)
        X = np.arange(0, newWidth - xStep, xStep)
        features[i] = np.empty(shape=(int(newHeight / HeightStep) - 1, int(newWidth / WidthStep) - 1, 1000), dtype=np.float)
        for row, y in enumerate(Y):
            for column, x in enumerate(X):
                print(y, row, column, x)
                features[i][row, column, :] = model.predict( np.expand_dims( preprocess_input( np.asarray(Image.fromarray(np.asarray(im)[y : y + winStep, x : x + winStep]).resize(model.input_shape[1:3], Image.ANTIALIAS), dtype=np.float32) ), axis=0 ) )

    # PATCH PREDICT
    predictions = model2.predict(np.concatenate(features[0]), batch_size=1000, verbose=0) # right for only one image

    # EVALUATE
    k = 0
    for i, feature in enumerate(features):
        height, width, _ = feature.shape
        # The marginal data of the predicted count matrix is 0 after apply MRF,
        # so first extending the predicted count matrix by copy marginal data.
        p = predictions[k : k + height * width].reshape(width, height).astype(np.float32).T
        p = np.asarray(ImageOps.expand(Image.fromarray(p), border=1)) # add black border
#        p = cv2.copyMakeBorder( p,
#                                top=1,
#                                bottom=1,
#                                left=1,
#                                right=1,
#                                borderType= cv2.BORDER_REPLICATE)
        k += height * width
        # apply MRF
        ############################ TO DO (capture reference and not copy buffer) ##############
        p = np.fromiter(MRF(p.ctypes.data_as(ptr_im), ctypes.byref(ctypes.c_int(height)), ctypes.byref(ctypes.c_int(width)), MRFParams.ctypes.data_as(ptr_flt)) , dtype=np.int32, count=(height+1)*(width+1)).reshape(height+1, width+1)
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
    #                                         ctypes.c_int(len(predictions)),
    #                                         MRFParams),
    #                        dtype=np.float64, count=len(predictions)
    #                        )

    # debug output
    finalcount.savetxt( fname="final_count.csv",
                        delimiter=",",
                        newline="\n",
                        fmt="%d")