import cv2
import numpy as np
import random
import  help
     
sift = cv2.xfeatures2d.SIFT_create()    

##----------------------------------------------Assignment 2 portion-----------------------------------------------------------------###
# def startup(image1,image2):
#     img=image1
#     img = cv2.imread(image1)
#     img2 = cv2.imread(image2)
#     gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#     kp,des=calculate_kp_des(gray)
#     kp2,des2=calculate_kp_des(gray2)
#     return img,img2,gray,gray2,kp,des,kp2,des2

def startup(image1):
    img = cv2.imread(image1)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp,des=calculate_kp_des(gray)
    return img,gray,kp,des

def startup2(image1):
    img = image1
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp,des=calculate_kp_des(gray)
    return img,gray,kp,des

def calculate_kp_des(img):
    kp, des = sift.detectAndCompute(img,None)
    return kp,des

def draw_kp(name,imgtemp,kp):
    imgx=cv2.drawKeypoints(imgtemp,kp,imgtemp,color=(0,255,0))

def draw_matches(kp,kp2,des,des2,img,img2):
    bf = cv2.BFMatcher( crossCheck=True)
    matches=bf.match(des,des2)
    matches=sorted(matches,key=lambda x:x.distance)
    imgx2 = cv2.drawMatches(img,kp,img2,kp2,matches, img2)
    return matches

#------------------Assignment 3 starts-----------------------------------------------------------------------------------####

def computeInnerCount(H,matches,inlierThreshold):
    print("")

def getRandromMatches(matches):
    random_choice=[]
    for i in range(0,4):
        random_choice.append(random.choice (matches))
    return random_choice

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

### Given H find new x2 , y2
def newAndOldDestPoints(matches,H,kp,kp2):
    src_pts = np.float32( [kp[match.queryIdx].pt for match in matches])
    dst_pts = np.float32( [kp2[match.trainIdx].pt for match in matches])
    new_des_pts=[]
    for i in src_pts:
        x1,y1=i
        x2,y2=project(x1,y1,H)

        new_des_pts.append((x2,y2))
    return np.array(new_des_pts),dst_pts

### Calculate the distance between new x2,y2 and orignal x2,y2    
def distance_calculate(dst,ndst):
    dist_mat=[]
    for i in range(len(dst)):
        a=((dst[i][0]-ndst[i][0])**2+(dst[i][1]-ndst[i][1])**2)**.5
        dist_mat.append(a)
    return dist_mat

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
        new_des_pts,dst_pts=newAndOldDestPoints(matches,H,kp,kp2)
        x=[]
        dist_mat=distance_calculate(dst_pts,new_des_pts)
        for i in range(0,len(dist_mat)) :
            if dist_mat[i]<inlierThreshold:
                x.append(i)        
        homocount.append((len(x),x))
    homocount.sort( reverse=True)
    return homocount[0][1]

def stichedImage():
    print("")

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



def main():
 numIterations=1000
 inlierThreshold=.75
 image1="project_images/Rainier6.png"
 image2="project_images/Rainier2.png"
 image3="project_images/Rainier3.png"
 image4="project_images/Rainier4.png"
 image5="project_images/Rainier5.png"
 image6="project_images/Rainier1.png"
 images=[image1,image2,image3,image4,image5,image6]
 emptyCanvas = cv2.imread(image1)

 kpArray=[]

 for i in range(1,6):
     img,gray,kp,des=startup(images[i])
     kpArray.append((img,gray,kp,des,i))


 # for i in range(0, 5):
 #     img, img2, gray, gray2, kp, des, kp2, des2 = startup(emptyCanvas, images[i + 1])
 #     draw_kp("image",gray,kp)
 #     draw_kp("image2",gray2,kp2)
 #     matches=draw_matches(kp,kp2,des,des2,img,img2)
 #
 # ##Applying Ransac and finding new matches with inliers
 #     homocount=Ransac(matches,numIterations,inlierThreshold,kp,kp2)
 #     new_matches,new_kp,new_kp2=getMatches(matches,homocount,kp,kp2)

 for picnum in range(0,len(kpArray)):
    img, gray, kp, des = startup2(emptyCanvas)
    tempArray=[]
    for i in range(0,len(kpArray)):
        matches=draw_matches(kp, kpArray[i][2], des, kpArray[i][3], img, kpArray[i][0])
        homocount=Ransac(matches,numIterations,inlierThreshold,kp,kpArray[i][2])
        new_matches,new_kp,new_kp2=getMatches(matches,homocount,kp,kpArray[i][2])
        tempArray.append((len(new_matches),new_matches,kpArray[i]))
    sorted(tempArray, key=lambda tup: tup[0],reverse=True)
##Finding Homography using new new matches
    img2, gray2, kp2, des2,n =tempArray[0][2]
    print(n)
    kpArray.remove(tempArray[0][2])
    print(len(kpArray))
    Hnew=findMyHomography(tempArray[0][1],kp,kp2)
    # inverse of matrix
    inverse = np.linalg.inv(Hnew)
    # imgx3 = cv2.drawMatches(img,kp,img2,kp2,new_matches, img2)
    ##Calculating new Projection Points
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
    emptyCanvas=np.empty((ymax,xmax,3),np.uint8)
    a,b,c=img2.shape

    try:
        for i in range(0,img.shape[1]):
            for j in range(0,img.shape[0]):
                emptyCanvas[ystart+j][xstart+i]=img[j][i]
    except:
        print("Error Copying in Image 1")

    for i in range(emptyCanvas.shape[1]):
        for j in range(emptyCanvas.shape[0]):
            x2,y2=project(i-xstart,j-ystart,Hnew)
            if x2>xImg2Old and x2<xImg2New and y2>yImg2Old and y2<yImg2New :
               emptyCanvas[j][i]=cv2.getRectSubPix(img2,(1,1),(x2,y2))
    # emptyCanvas=cv2.addWeighted(emptyCanvas,0.5,tempImage3,0.5,0)
    aaa=str(picnum)+".jpg"
    cv2.imwrite(aaa, np.uint8(emptyCanvas))
    cv2.imshow(aaa, np.uint8(emptyCanvas))

main()

# cv2.imshow("image2", np.uint8(emptyCanvas))

cv2.waitKey(0)
cv2.destroyAllWindows()