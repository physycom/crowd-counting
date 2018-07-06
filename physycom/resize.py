import sys
import cv2

if len(sys.argv) < 2:
  print("Usage :",sys.argv[0],"path/to/img")
  exit(1)

tiny_w = 1024
# Resize images
file=sys.argv[1]
im = cv2.imread(file)
tiny_file=file.split(".")[0] + "_tiny.jpg"
h, w, c = im.shape
ratio = h/float(w)
tiny_h = int(ratio * tiny_w)
print("Resizing :",file,"@",tiny_w,"x",tiny_h)
tiny = cv2.resize(im, (tiny_w, tiny_h))
cv2.imwrite(tiny_file, tiny)
