import sys
import cv2

if len(sys.argv) < 2:
  print("Usage :",sys.argv[0],"path/to/img")
  exit(1)

# Grayscaling images
file = sys.argv[1]
im = cv2.imread(file)
gray_file = file.split(".")[0] + "_gs.jpg"
print("Grayscaling :",file)
gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite(gray_file, gray_image)
