from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", 
    help = "paht to video")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else: 
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    
    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = frame.copy()
 
	# detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
        padding=(8, 8), scale=1.05)
 
	# draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

 

    cv2.imshow("Ped Detect", frame)
    if(cv2.waitKey(1) & 0xFF == ord("q")):
        break
camera.release()
cv2.destroyAllWindows()