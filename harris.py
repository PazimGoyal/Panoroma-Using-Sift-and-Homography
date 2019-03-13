import cv2
import numpy as np
     
img = cv2.imread('project_images/Boxes.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     
sift = cv2.xfeatures2d.SIFT_create()     
kp, des = sift.detectAndCompute(img,None)
img=cv2.drawKeypoints(gray,kp,img,color=(0,255,0), flags=0)
cv2.imshow("f",img)
