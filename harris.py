import cv2
import numpy as np
     
sift = cv2.xfeatures2d.SIFT_create()    

def calculate_kp_des(img):
    kp, des = sift.detectAndCompute(img,None)
    return kp,des


def draw_kp(name,img,kp):
    img=cv2.drawKeypoints(img,kp,img,color=(0,255,0), flags=0)
    cv2.imshow(name,img)
    



def main():
    img = cv2.imread('project_images/ND1.png')
    img2 = cv2.imread('project_images/ND2.png')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    kp,des=calculate_kp_des(gray)
    draw_kp("image",img,kp)

    kp2,des2=calculate_kp_des(gray2)
    draw_kp("image2",img2,kp2)
 
    

main()
