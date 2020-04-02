import random
import math
import  os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import math
from scipy import signal  # 滤波等
from scipy.signal import medfilt
from scipy import interpolate
from libs.tony_beltramelli_detect_peaks import detect_peaks


def compute_distance(point_1, point_2):
    """ 计算两个点之间的距离"""
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    if point_1.shape != point_2.shape:
        raise ValueError("shape of image must be equal to reference")
    point_1 = point_1.astype(np.float32)
    point_2 = point_2.astype(np.float32)
    distance = np.mean(np.square(point_1-point_2))
    return distance
def testli11111(oth,image,ys,image_name):
    for n in range(len(oth)) :
        # if oth[n]!=None :
        # print('oth:',oth[n])
            cv2.circle(image, (oth[n][1],oth[n][0]), 1 , ys )

    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, 1000, 1000)
    cv2.imshow(image_name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image
def testl_line(pt1,oth,image,ys=(255,0,0),image_name='img1'):
    for n in range(len(oth)) :
        # if oth[n]!=None :
        # print('oth:',oth[n])
        #     cv2.circle(image, (oth[n][1],oth[n][0]), 1 , ys )
              cv2.line(image,pt1,(oth[n][1],oth[n][0]),ys)
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, 1000, 1000)
    cv2.imshow(image_name,image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

def showImage(image, method = 'plt'):
    if method == 'plt':
        try:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        except cv2.error:
            plt.imshow(image)
            plt.show()

    else:
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def get_maxmin(point_list):
    max_x = max(np.array(point_list)[:,:,0])
    max_y = max(np.array(point_list)[:,:,1])
    min_x = min(np.array(point_list)[:,:,0])
    min_y = min(np.array(point_list)[:,:,1])
    return max_x,max_y,min_x,min_y
def getFw_list(all_contours):
    Fw  =[]
    for i in all_contours:
        max_x_1, max_y_1, min_x_1, min_y_1 = get_maxmin(i)
        Fw.append([max_x_1, max_y_1, min_x_1, min_y_1])
    return Fw
def getDX_fw(contour_FW1,contour_FW2):
    x1 = contour_FW1[0] > contour_FW2[0]
    y1 = contour_FW1[1] > contour_FW2[1]
    x2 = contour_FW1[2] < contour_FW2[2]
    y2 = contour_FW1[3] < contour_FW2[3]
    return x1,y1,x2,y2

def geiFw_Info(all_contours, Fw_list, save_path, image):
    # 所有的contours个数
    E = len(all_contours)
    nameNum = 0
    file_names = []
    del_file = []
    for i in range(E):
        new_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        nameNum += 1
        fwli = get_maxmin(all_contours[i])
        bihua_FW_li = []
        bihua_FW_li.append(all_contours[i])
        houxu_fw = []

        for j in range(len(Fw_list)):

            max_x, max_y, min_x, min_y = getDX_fw(fwli, Fw_list[j])

            if max_x == True and max_y == True and min_x == True and min_y == True:
                houxu_fw.append(all_contours[j])
                del_file.append(j + 1)

        filename = save_path + str(nameNum) + '.jpg'

        cv2.drawContours(new_img, bihua_FW_li, -1, 255, -1)
        if len(houxu_fw) != 0:
            cv2.drawContours(new_img, houxu_fw, -1, 0, -1)
        cv2.imwrite(filename, new_img)
        file_names.append(filename)
    return file_names

def threshold_image(image, min_p = 200, max_p = 255):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayscaled, min_p, max_p, cv2.THRESH_BINARY)
    # cv2.imshow('111',threshold)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return threshold


def getSkeleton (image):
    # Applies a skeletonization algorithm, thus, transforming all the strokes inside the sketch into one pixel width lines
    threshold = threshold_image(image)
    ceshi = []
    threshold = cv2.bitwise_not(threshold)
    threshold[threshold == 255] = 1
    skeleton = skeletonize(threshold)
    skeleton = img_as_ubyte(skeleton)
    h , w = skeleton.shape
    for i in range(h):
        for j in range(w):
            if skeleton[i,j]==255:
                ceshi.append([i,j])
    return skeleton,ceshi

def get_NC(n_list):
    nu = 0
    for i in range(len(n_list)-1):
        nu  = n_list[i+1]-n_list[i]
        if i==7:
            nu = n_list[len(n_list)-1] - n_list[0]
    nu = nu/8
    return nu

 # 查询8邻域内黑色坐标位置
def scan_8_pixel_neighbourhood(skeleton_image, pixel):
    """
    :param skeleton_image: skeleton image
    :param pixel: a tuple of the type (x, y)
    :return: a matrix indicating the indexes of the neighbouring pixels of the input pixel
    """
    if inside_image(pixel, skeleton_image):
        skeleton_image = skeleton_image.copy()
        neighbourhood = skeleton_image[pixel[1] - 1: pixel[1] + 2, pixel[0] - 1: pixel[0] + 2]
        neighbourhood[1,1] = 0
        neighbours = np.argwhere(neighbourhood)
        return neighbours
    else:
        return []


def inside_image(pixel, image):
    """Checks whether a pixel is inside the image space"""
    h, w = image.shape
    if (pixel[1] - 1 >= 0) and (pixel[1] + 1 <= h - 1) and (pixel[0] - 1 >= 0) and (pixel[0] + 1 <= w - 1):
        return True
    else:
        return False

def get_N():
    while True:
        n = random.randint(-2.2)
        if n!=0:
            return n

def neighbours_x_y(x,y):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    # if  x=10.y=10   3点钟方向 逆时针 点位

    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
     # p0        p1            p2               p3          p4              p5          p6             p7
    return [[x,y1] ,[x_1,y1], [x_1,y],  [x_1,y_1], [x,y_1] , [x1,y_1],  [x1,y], [x1,y1]]


def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    # if  x=10.y=10   3点钟方向 逆时针 点位
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
                 # p0        p1            p2               p3          p4              p5          p6             p7
    return [img[x][y1] ,img[x_1][y1], img[x_1][y],  img[x_1][y_1], img[x][y_1] , img[x1][y_1],  img[x1][y], img[x1][y1]]

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
    #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1]==e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: #线段在射线上边
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: #线段在射线下边
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
        return False
    if s_poi[0]<poi[0] and e_poi[1]<poi[1]: #线段在射线左边
        return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) #求交
    if xseg<poi[0]: #交点在射线起点的左侧
        return False
    return True  #排除上述情况之后

def Translate(X,Y,angle,distance):                #defines function
    # 0 degrees = North, 90 = East, 180 = South, 270 = West
    dY = distance*math.cos\
        (math.radians(angle))   #change in y
    dX = distance*math.sin(math.radians(angle))   #change in x
    Xfinal = X + dX
    Yfinal = Y + dY
    return Xfinal, Yfinal

save_path = 'images/img'
max_x,max_y = 0,0
min_x,min_y = 0,0


img1 = cv2.imread('images/test2.png', cv2.IMREAD_GRAYSCALE)  # 读取图像

if len(img1.shape) > 2:
    h1, w1, n = img1.shape
    img2 = img1[::, 2]
else:
    h1, w1 = img1.shape
    img2 = img1



threshold = cv2.bitwise_not(img2)
threshold[threshold<200]=0
threshold[threshold>1]=255
contours, cnt = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 提取轮廓 contours

save_path = 'images/img'
max_x,max_y = 0,0
min_x,min_y = 0,0

Fw_list =getFw_list(contours)
files_name = geiFw_Info(contours,Fw_list,save_path,img1)
print(files_name)
img = cv2.imread(files_name[0],0)
showImage(img)
white = []
for i in range(h1):
    for j in range(w1):
        if img[i,j]==255:
            white.append([i,j])

bofeng_0 = []
bofeng_1 = []
bofeng_2 = []
bofeng_3 = []
bofeng_4 = []
# cv2.imread('images/tes')

test_iamge = cv2.imread('images/test2.png')  # 读取图像
edge_list= []
threshold = cv2.bitwise_not(img)
threshold[threshold<200]=0
threshold[threshold>1]=255

for i in white:
    n = neighbours(i[0],i[1],threshold)
    if len(np.argwhere(np.array(n)>=10))>=3:
        edge_list.append(i)

testli11111([[208, 211]], test_iamge, (255, 0, 0), 'img')

# showImage(img)

# white = [[221, 179]]  211, 208
js = 0
for poit in white:
    poit= [208, 211]
    bodong =[]
    Dis = []
    print(poit)
    # testli11111([poit], test_iamge, (255, 0, 0),'img'+str(poit))
    for i in range(0,360):
        # print('角度是:',i)
        # test_iamge = cv2.imread('images/test2.jpg')
        for j in range(len(img)):
            sx_x,sx_y = Translate(poit[0],poit[1],i,j)
            if img[int(sx_x),int(sx_y)]==0:
                # print('黑色的点位是: ',[sx_x,sx_y])
                bodong.append([int(sx_x),int(sx_y)])
                jl = compute_distance((poit[0],poit[1]),(int(sx_x),int(sx_y)))
                Dis.append(jl)
                # cv2.line(test_iamge,(poit[1],poit[0]),(int(sx_y),int(sx_x)),(255,0,0))
                # cv2.destroyAllWindows()
                break
        # cv2.imshow('11', test_iamge)
        # cv2.waitKey(0)

    js +=1
    # testli11111(bodong, test_iamge, (0, 255, 0),'img'+str(poit))
    # print(bodong)
    print(Dis)
    # showImage(img)
    # bodogn = np.array(bodong)
    # Dis = medfilt(Dis)
    # Dis = np.array(Dis)
    # lb = signal.medfilt(Dis)
    # print(lb)



    # x = np.linspace(0,len(Dis),len(Dis))           # 设置x长度
    # x_new = np.linspace(0,len(Dis),len(Dis))    # 10倍插值
    # kind = "quadratic"                 # "quadratic","cubic" 为2阶、3阶B样条曲线插值
    # f = interpolate.interp1d(x,x,kind=kind)
    # DIS = f(x_new)
    # plt.plot(x_new,Dis) # 用插值后的绘制图像
    # plt.xlabel('x')      # x轴标签
    # plt.ylabel('y')      # y轴标签
    # #plt.legend('')     # 标签
    # plt.show()
    indexes = detect_peaks(Dis, 0.6)
    print(indexes)
    quchu_list = []
    for index in range(len(indexes) - 1):
        if abs(indexes[index] - indexes[index + 1]) <= 10:
            quchu_list.append(indexes[index])
    indexes = list(indexes)
    print(quchu_list)
    for quchu in quchu_list:
        indexes.remove(quchu)
    print(indexes)
    print('indexes:', indexes)
    print('峰值是：', len(indexes))
    # test_iamge = cv2.imread('images/test2.jpg')
    # testl_line((poit[1],poit[0]),bodong,test_iamge)
    # testli11111([poit],test_iamge,(255,0,0),'img1')
    # k_mean = np.mean(Dis)
    # num_peak_3 = signal.find_peaks(Dis,threshold=1.5) #distance表极大值点的距离至少大于等于10个水平单位

    # indexes = signal.argrelextrema(Dis,np.less)
    # print(indexes)
    # print(k_mean)
    # print('原始点:',poit)
    # Dis = np.array(Dis) * -1
    # indexes = signal.find_peaks(Dis)
    # indexes = indexes[0]

    bofeng = len(indexes)
    # print(bofeng)
    if bofeng == 0:
        bofeng_0.append([poit[0], poit[1]])
    if bofeng==1 :
        bofeng_1.append([poit[0],poit[1]])
    elif bofeng==2:
        bofeng_2.append([poit[0],poit[1]])
    elif bofeng>=3:
        bofeng_3.append([poit[0],poit[1]])
    # elif bofeng>=4:
    #     bofeng_4.append([poit[0],poit[1]])
    # testli11111([[poit[0], poit[1]]], test_iamge, (0, 0, 0))
    break
# print(len(bofeng_3))
# for i in range(len(bofeng_3)-1):
#     if bofeng_3[i]==bofeng_3[i+1]:
#         bofeng_3.remove(bofeng_3[i+1])
# print(len(bofeng_3))


# ce_li = []
# for i in range(len(bofeng_3)-2):
#     if  bofeng_3[i] < bofeng_3[i+1] or \
#             bofeng_3[i] < bofeng_3[i+2] or \
#             bofeng_3[i] < bofeng_3[i-2] or \
#             bofeng_3[i] < bofeng_3[i-1]:
#         ce_li.append(bofeng_3[i])
# print('white',ce_li)

# testli11111(ce_li,test_iamge,(0,0,255),'img5')
# print(bofeng_1)
# print(len(bofeng_1))
# print(bofeng_2)
# print(len(bofeng_2))
# print(bofeng_3)
# print(len(bofeng_3))

# print('all_point',len(white))
# testli11111(bofeng_0,test_iamge,(255,255,0),'img1')
# testli11111(bofeng_1,test_iamge,(255,0,0),'img2')
# testli11111(bofeng_2,test_iamge,(0,255,0) ,'img3')
# testli11111(bofeng_3,test_iamge,(0,255,255) ,'img4')
# testli11111(bofeng_4,test_iamge,(0,0,255),'img5')