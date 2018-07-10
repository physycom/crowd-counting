import cv2
import scipy.io as sci
import numpy as np

features = sci.loadmat(r'E:\Alessandro\Codice\crowd-counting\physycom\count\chiara\perfetta_1_tiny.phase_0.mat')
(w, h, mille) = features['features'].shape
patches  = sci.loadmat(r'E:\Alessandro\Codice\crowd-counting\physycom\count\chiara\perfetta_1_tiny.phase_1.mat')

pred = np.zeros((w+1,h+1))
pred[:-1,:-1] = patches['predictions'].reshape(w,h)

pred_min = np.min(pred)
pred_max = np.max(pred)
scaled_pred = np.uint8( np.concatenate( [ np.concatenate([np.full((50,50), int(255*val/(pred_max-pred_min))) for val in row], axis=1) for row in pred ] ))

scaled_pred = cv2.applyColorMap(scaled_pred, cv2.COLORMAP_JET)

image = cv2.imread(r'E:\Alessandro\Codice\crowd-counting\physycom\count\test\venice_012345.jpg')
overlay = cv2.addWeighted(image, 1, scaled_pred, 0.7, 0)

#cv2.imshow("image",image)
#cv2.imshow("patches",scaled_pred)
#cv2.imshow("heatmap",overlay)

compare = np.concatenate((image, scaled_pred, overlay), axis=1)
compare = cv2.resize(compare, (0,0), fx=0.6, fy=0.6)
cv2.imshow("compare",compare)
