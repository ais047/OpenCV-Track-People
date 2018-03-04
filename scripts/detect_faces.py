from __future__ import print_function
from pyimagesearch.facedetector import FaceDetector 
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
    help = "path to face cascade")
ap.add_argument("-i", "--image", required = True,
    help = "path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fd = FaceDetector(args["face"])
faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
    minSize = (30,30))
print("I found {} faces".format(len(faceRects)))
for (x,y,w,h) in faceRects:
        cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces", image)
cv2.waitKey(0)