import cv2
import numpy as np
import random
sift = cv2.xfeatures2d.SIFT_create()    
np.seterr(divide="ignore",invalid="ignore")
##----------------------------------------------Assignment 2 portion-----------------------------------------------------------------###
# this function takes two images and return keypoints and descriptors for those images
def startup(image1,image2):
    img=image1
    img2 = cv2.imread(image2)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    kp,des=calculate_kp_des(gray,img)
    kp2,des2=calculate_kp_des(gray2,img2)
    return img,img2,gray,gray2,kp,des,kp2,des2

# Calculating keypoins using Harris Corner and descriptor using sift
def calculate_kp_des(img,org):
    points = cv2.cornerHarris(img, 2, 3, 0.04)
    tempImage = org.copy()
    tempImage[points > 0.01 * points.max()] = [0, 0, 255]
    kp = np.argwhere(points > 0.01 * points.max())
    kp = [cv2.KeyPoint(x[1], x[0], 1) for x in kp]
    kp,des = sift.compute(img,kp)
    return kp,des

#This method simply draws keypoints  on image
def draw_kp(name,imgtemp,kp):
    imgx=cv2.drawKeypoints(imgtemp,kp,imgtemp,color=(0,255,0))
    cv2.imwrite(name, imgx)


# This method simply draw all the matches on the images
def draw_matches(kp,kp2,des,des2,img,img2,name):
    bf = cv2.BFMatcher( crossCheck=True)
    matches=bf.match(des,des2)
    matches=sorted(matches,key=lambda x:x.distance)
    imgx2 = cv2.drawMatches(img,kp,img2,kp2,matches, img2)
    cv2.imwrite(name,imgx2)
    return matches



#------------------Assignment 3 starts-----------------------------------------------------------------------------------####






# this returns four random matches out of all the matches
def getRandromMatches(matches):
    random_choice=[]
    for i in range(0,4):
        random_choice.append(random.choice (matches))
    return random_choice

#  this finds the homography using matches
def findMyHomography(f_matches,kp,kp2):
    src_pts = np.float32( [kp[match.queryIdx].pt for match in f_matches])
    dst_pts = np.float32( [kp2[match.trainIdx].pt for match in f_matches])
    H, mask = cv2.findHomography(src_pts, dst_pts,0)
    return H

# Find the new  Matches with INLIERS

def getMatches(matches,homocount,kp,kp2):
    new_matches=[]
    new_kp=[]
    new_kp2=[]
    for i in homocount:
        new_matches.append(matches[i])
        new_kp.append(kp[matches[i].queryIdx])
        new_kp2.append(kp2[matches[i].trainIdx])
    return new_matches,new_kp,new_kp2


### Calculate the distance between new x2,y2 and orignal x2,y2
def computeInlierCount(H,matches,inlierThreshold,kp,kp2):
    src_pts = np.float32([kp[match.queryIdx].pt for match in matches])
    dst = np.float32([kp2[match.trainIdx].pt for match in matches])
    x=[]

    for i in range(0,len(src_pts)):
        x1, y1 = src_pts[i]
        x2, y2 = project(x1, y1, H)
        a=((dst[i][0]-x2)**2+(dst[i][1]-y2)**2)**.5
        if a < inlierThreshold:
            x.append(i)
    return x


def project(x1,y1,H):
    try:
        temp = np.array([x1, y1, 1], np.float32)
        u, v, w = np.dot(H, temp)
        x2, y2 = u / w, v / w
        return x2,y2
    except:
        return x2, y2



def Ransac(matches,numIterations,inlierThreshold,kp,kp2):
    homocount=[]
    for j in range(0,numIterations):
        f_matches=getRandromMatches(matches)
        H=findMyHomography(f_matches,kp,kp2)
        x=computeInlierCount(H,matches,inlierThreshold,kp,kp2)
        homocount.append((len(x),x))
    homocount.sort( reverse=True)
    return homocount[0][1]


# Stiching Function
def stitch(img, img2,Hnew,inverse):
    co_ordsImg1,old_cods=co_ordinates(img,img2,inverse)
    co_ordsImg2=[]
    for i in range(0,4):
        x2,y2=project(old_cods[i][0],old_cods[i][1],inverse)
        co_ordsImg2.append((x2,y2))
    xImg1Old,yImg1Old=co_ordsImg1[0]
    xImg1New,yImg1New=co_ordsImg1[3]

    xstart=0
    ystart=0
    xmax=img.shape[1]
    ymax=img.shape[0]



    for i in range(0,4):
        if xImg1Old > co_ordsImg2[i][0] :
            xmax=round(xmax+xImg1Old-co_ordsImg2[i][0],0)
            xstart=round(xImg1Old - co_ordsImg2[i][0],0)
        if  xImg1New<co_ordsImg2[i][0] and co_ordsImg2[i][0]>xmax:
            xmax=round(co_ordsImg2[i][0],0)
        if yImg1Old>co_ordsImg2[i][1]:
            ymax=round(ymax+yImg1Old-co_ordsImg2[i][1],0)
            ystart=round(yImg1Old-co_ordsImg2[i][1],0)
        if   yImg1New<co_ordsImg2[i][1]and co_ordsImg2[i][1]>ymax:
            ymax=round(co_ordsImg2[i][1],0)
    # print("--------------------------------")

    xImg2Old, yImg2Old = old_cods[0][:2]
    xImg2New, yImg2New = old_cods[3][:2]
    xstart=int(round(xstart,0))
    ystart = int(round(ystart, 0))
    ymax=int(ymax)
    xmax=int(xmax)
    emptyCanvas=np.empty((ymax,xmax,3))
    print("Canvas Created")


    try:
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                emptyCanvas[ystart+j][xstart+i]=img[j][i]
    except:
        print("Error Copying in Image 1")
    print("Image 1 copied")

    for i in range(emptyCanvas.shape[1]):
        for j in range(emptyCanvas.shape[0]):
            x2,y2=project(i-xstart,j-ystart,Hnew)
            if x2>xImg2Old and x2<xImg2New and y2>yImg2Old and y2<yImg2New :
               emptyCanvas[j][i]=cv2.getRectSubPix(img2,(1,1),(x2,y2))



    emptyCanvas=np.uint8(emptyCanvas)
    return emptyCanvas


def co_ordinates(img,img2,inverse):
    co_ordsImg1 = []
    xb0, yb0 = 0, 0
    xb1, yb0 = img.shape[1], 0
    xb0, yb1 = 0, img.shape[0]
    xb1, yb1 = img.shape[1], img.shape[0]
    co_ordsImg1.append((xb0, yb0))
    co_ordsImg1.append((xb1, yb0))
    co_ordsImg1.append((xb0, yb1))
    co_ordsImg1.append((xb1, yb1))


    #### Finding the four co-ordinatess  of image 2

    old_cods=[]
    xa0,ya0=0,0
    xa1,ya0=img2.shape[1],0
    xa0,ya1=0,img2.shape[0]
    xa1,ya1=img2.shape[1],img2.shape[0]
    old_cods.append(( xa0,ya0))
    old_cods.append(( xa1,ya0))
    old_cods.append(( xa0,ya1))
    old_cods.append(( xa1,ya1))

    ####    new Co-ordianates of image 2
    return np.array(co_ordsImg1),old_cods

def boxes():
    image1 = "project_images/Boxes.png"
    im = cv2.imread(image1)
    gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    kp,des=calculate_kp_des(gray,im)
    draw_kp("1a.png",im,kp)



def main():
 boxes()
 numIterations=500
 inlierThreshold=.75  
 image1="project_images/Rainier1.png"
 image2="project_images/Rainier2.png"
 image3="project_images/Rainier3.png"
 image4="project_images/Rainier4.png"
 image5="project_images/Rainier5.png"
 image6="project_images/Rainier6.png"
## image1="project_images/image1.jpg"
## image2="project_images/image2.jpg"
## image3="project_images/image3.jpg"

 images=[image1,image2,image3,image4,image5,image6]
 emptyCanvas = cv2.imread(image1)
 kpArray=[]

 for i in range(0, 5):
    print("Computing canvas and Image ",(i+1))
    calculating="Image{}.png".format(i+1)
    img, img2, gray, gray2, kp, des, kp2, des2 = startup(emptyCanvas, images[i + 1])
    draw_kp("1a-"+calculating,gray,kp)
    draw_kp("1b-"+calculating,gray2,kp2)
    matches=draw_matches(kp,kp2,des,des2,gray,gray2,"2-"+calculating)
    #Applying Ransac and finding new matches with inliers
    homocount=Ransac(matches,numIterations,inlierThreshold,kp,kp2)
    new_matches,new_kp,new_kp2=getMatches(matches,homocount,kp,kp2)
    Hnew=findMyHomography(new_matches,kp,kp2)
    # inverse of matrix
    inverse = np.linalg.inv(Hnew)
    #New Image uising new Matches
    imtemp=img2.copy()
    imgx3 = cv2.drawMatches(img,kp,img2,kp2,new_matches,imtemp)
    cv2.imwrite("3-"+calculating,imgx3)

    #Stiching Two images
    emptyCanvas= stitch(img,img2,Hnew,inverse)

    #CODE FOR BLENDING THE IMAGES

    # for i in range(emptyCanvas.shape[1]):
    #     for j in range(emptyCanvas.shape[0]):
    #         temp_array = np.array((emptyCanvas[j][i], emptyCanvas2[j][i]))
    #         if emptyCanvas[j][i][0]==0 and emptyCanvas2[j][i][0]==0:
    #             emptyCanvas[j][i] = ([0,0,0])
    #         elif emptyCanvas[j][i][0]==0 :
    #             emptyCanvas[j][i] = emptyCanvas2[j][i]
    #         elif emptyCanvas2[j][i][0] ==0:
    #             emptyCanvas[j][i]=emptyCanvas[j][i]
    #         else:
    #             emptyCanvas[j][i] = (0.5*emptyCanvas[j][i])+(0.5*emptyCanvas2[j][i])
    cv2.imwrite("4-"+calculating,emptyCanvas)
 cv2.imshow("4-"+calculating, emptyCanvas)
main()
cv2.waitKey(0)
