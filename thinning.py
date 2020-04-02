import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import cv2 as cv
"load image data"
from PIL import Image

# Img_Original = io.imread('./data/4E4c.bmp')
# Img_Original =  cv.imread( 'images/test4.bmp'  )
Img_Original =  cv.imread( 'images/test2.png')
print(Img_Original.shape)

Img_Original = cv.cvtColor(Img_Original, cv.COLOR_RGB2GRAY);
# cv.imshow('111',Img_Original2)
# cv.waitKey(0)
# cv.destroyAllWindows()
# Img_Original = Img_Original[2]
# print(Img_Original)
# Img_Original = Image.open('./data/4EAC.png')
# Img_Original = Image.fromarray(np.uint8(Img_Original)*20)


"Convert gray images to binary images using Otsu's method"
# from skimage.filter import threshold_otsu
from  skimage.filters import threshold_otsu
Otsu_Threshold = threshold_otsu(Img_Original)   
BW_Original = Img_Original < Otsu_Threshold    # must set object region as 1, background region as 0 !
print(1)
def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def Skeleton(image):
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned


def Translate(X, Y, angle, distance):  # defines function
    # 0 degrees = North, 90 = East, 180 = South, 270 = West
    dY = distance * math.cos \
        (math.radians(angle))  # change in y
    dX = distance * math.sin(math.radians(angle))  # change in x
    Xfinal = X + dX
    Yfinal = Y + dY
    return Xfinal, Yfinal

"Apply the algorithm on images"
BW_Skeleton = Skeleton(BW_Original)
# BW_Skeleton = BW_Original
"Display the results"
fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.ravel()
ax1.imshow(BW_Original, cmap=plt.cm.gray)
ax1.set_title('Original binary image')
ax1.axis('off')
ax2.imshow(BW_Skeleton, cmap=plt.cm.gray)
ax2.set_title('Skeleton of the image')
ax2.axis('off')
plt.show()
import cv2
def testli11111(oth,image,ys):
    for n in range(len(oth)) :
        if oth[n]!=None:
        # print('oth:',oth[n])
            cv2.circle(image, (oth[n][0],oth[n][1]), 1 , ys )
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1000, 1000)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    return image

def get_NC(n_list):
    nu = 0
    for i in range(len(n_list)-1):
        nu  = n_list[i+1]-n_list[i]
    nu = nu/8
    print(nu)
    return nu
def getImage_jd(image):
    jiaodian_list=[]
    line_list = []
    startorend_list = []
    h,w = image.shape
    print(h,w)
    for i in range(h):
        for j in range(w):
            if image[i,j]==255:
                n = neighbours(j,i,image)
                back = get_NC(n)
                if back>2:
                     testli11111([[j, i]], img, (255, 0, 0))
    return jiaodian_list

img = Img_Original.copy()
BW_Skeleton = BW_Skeleton*255
getImage_jd(BW_Skeleton)