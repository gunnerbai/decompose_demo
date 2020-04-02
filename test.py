import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



x_list = [[156, 166], [156, 164], [157, 166], [156, 166], [157, 164], [158, 165], [137, 166], [139, 165], [139, 166],
                 [139, 167], [156, 166], [156, 165], [158, 166], [137, 168], [136, 168], [138, 167], [138, 168], [137, 167],
                 [138, 166], [139, 166], [139, 168], [138, 166], [139, 166], [136, 169], [136, 168], [138, 167], [118, 181],
                 [117, 179], [119, 179]]

y_list = [[156, 166], [156, 164], [157, 166], [156, 166], [157, 164], [158, 165], [137, 166], [139, 165], [139, 166],
          [139, 167], [156, 166], [156, 165], [158, 166], [137, 168], [136, 168], [138, 167], [138, 168], [137, 167],
          [138, 166], [139, 166], [139, 168], [138, 166], [139, 166], [136, 169], [136, 168], [138, 167], [118, 181],
          [117, 179], [119, 179]]

import numpy as np
import scipy.stats
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块
from math import *
from sklearn.cluster import DBSCAN
import cv2

def testli11111(oth,image,ys):
    for n in range(len(oth)) :
        # print('oth:',oth[n])
        cv2.circle(image, (oth[n][0],oth[n][1]), 1 , ys )
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1000, 1000)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    return image
def testli222222(oth,image,ys):
    for n in range(len(oth)) :
        # print('oth:',oth[n])
        cv2.circle(image, (oth[n][1],oth[n][0]), 1 , ys )
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1000, 1000)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    return image



from numpy import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten

from numpy import *
import operator


# 设置数据集，以及每个数据对应的分类
def createDataSet():
    y1 = [(130, 111), (130, 112), (131, 113), (132, 114), (133, 115), (134, 116), (135, 117), (136, 118), (137, 119),
          (137, 120),
          (138, 121), (138, 122), (138, 123), (139, 124), (139, 125), (139, 126), (139, 127), (139, 128), (139, 129),
          (139, 130),
          (139, 131), (139, 132), (139, 133), (139, 134), (139, 135), (139, 136), (139, 137), (139, 138), (139, 139),
          (139, 140),
          (138, 141), (138, 142), (138, 143), (138, 144), (138, 145), (138, 146), (138, 147), (138, 148), (138, 149),
          (138, 150),
          (138, 151), (138, 152), (138, 153), (138, 154), (138, 155), (138, 156), (138, 157), (138, 158), (138, 159),
          (138, 160),
          (138, 161), (138, 162), (138, 163), (138, 164)]
    y2 = [(156, 142), (157, 143), (157, 144), (158, 145), (159, 146), (159, 147), (159, 148), (160, 149), (160, 150),
          (160, 151),
          (160, 152), (160, 153), (160, 154), (160, 155), (160, 156), (160, 157), (160, 158), (159, 159), (159, 160),
          (159, 161),
          (159, 162), (158, 163)]
    y3 = [(113, 154), (113, 155), (114, 156), (115, 157), (115, 158), (115, 159), (115, 160), (115, 161), (116, 162),
          (116, 163),
          (116, 164), (116, 165), (116, 166), (116, 167), (116, 168), (116, 169), (116, 170), (116, 171), (116, 172),
          (116, 173),
          (117, 174), (117, 175), (117, 176), (117, 177)]
    y4 = [(141, 166), (142, 166), (143, 166), (144, 166), (145, 166), (146, 166), (147, 166), (148, 166), (149, 166),
          (150, 166),
          (151, 166), (152, 166), (153, 166), (154, 166)]
    y5 = [(135, 170), (135, 171), (134, 172), (133, 173), (132, 174), (131, 175), (130, 176), (129, 177), (128, 177),
          (127, 177),
          (126, 178), (125, 178), (124, 178), (123, 178), (122, 179), (121, 179)]

    labels = ['A', 'B', 'C', 'D', 'E']
    return y1,y2,y3,y4,y5, labels

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

y1,y2,y3,y4,y5, labels1 = createDataSet()

group_list = []
labels_list = []
for i in y1:
    group_list.append(i)
    labels_list.append(labels1[0])
for i in y2:
    group_list.append(i)
    labels_list.append(labels1[1])
for i in y3:
    group_list.append(i)
    labels_list.append(labels1[2])
for i in y4:
    group_list.append(i)
    labels_list.append(labels1[3])
for i in y5:
    group_list.append(i)
    labels_list.append(labels1[4])

print(len(group_list))
print(len(labels_list))

labels = np.array(labels_list)
group = np.array(group_list)

img = cv2.imread('images/test2.jpg')



# group
# [[1.  1.1]
#  [1.  1. ]
#  [0.  0. ]
#  [0.  0.1]]

def classify(inX, dataSet, labels, k):
    # 获取行数
    dataSetSize = dataSet.shape[0]

    # 生成n个与训练样本数量对应的测试样本矩阵，计算测试样本与训练样本之间的距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # tile(inX, (dataSetSize, 1)) : 对inX，在列方向上重复dataSetSize次，在行方向上重复1次，生成dataSetSize * 1 的矩阵
    # 即
    # [[0, 0],
    #  [0, 0],
    #  [0, 0],
    #  [0, 0]]
    #
    # diffMat:
    # [[-1. - 1.1]
    #  [-1. - 1.]
    #  [0.   0.]
    #  [0. - 0.1]]
    # diffMat距离矩阵的每个距离值平方 (不平方也不太影响)
    sqDiffMat = power(diffMat, 2)
    # sqDiffMat = diffMat
    # [[1., 1.21],
    #  [1., 1.],
    #  [0., 0.],
    #  [0., 0.01]]

    # 将每一列的距离差方相加，生成测试样本和每一个训练样本的距离总和
    sqDistance = sqDiffMat.sum(axis=1)
    # 每行求和，变成nx1矩阵
    # sqDistance:
    # [2.21,2.,0.,0.01]

    # sqDistance矩阵的0.5次方 （不平方也太影响）
    distances = power(sqDistance, 0.5)
    # distances = sqDistance
    # [1.48660687, 1.41421356, 0., 0.1]

    # 将矩阵元素从小到大排序后的索引值（按照距离差，该索引值即为每一个测试样本的索引值）
    sortedInstance = distances.argsort()
    # [2, 3, 1, 0]
    classCount = {}

    # 将前k个距离最近的训练样本分类摘出来，计算每个分类的个数
    for i in range(k):
        # 获取测试样本分类
        voteIlabel = labels[sortedInstance[i]]
        # 该分类数量+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 将分类占据的个数降序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # reverse=True 表示降序  operator.itemgetter(1) 表示对dict中第一维的元素进行排序

    # 取出个数最多的那个分类
    # print(sortedClassCount)

    return sortedClassCount[0][0]


def compute_distance(point_1, point_2):
    """ 计算两个点之间的距离"""
    if point_1.shape != point_2.shape:
        raise ValueError("shape of image must be equal to reference")
    point_1 = point_1.astype(np.float32)
    point_2 = point_2.astype(np.float32)
    distance = np.mean(np.square(point_1-point_2))
    return distance
# j = 0
# juli_list = {}
# x_list = np.array(x_list)
# tmp_list = []
# k = 0
# x_list = np.sort(x_list)
# print(x_list)
# for i  in range(len(x_list)-1):
#
#     # print(k)
#     # print(i,i+1)
#     jl = compute_distance(x_list[i],x_list[i+1])
#     # print(x_list[i],x_list[i+1])
#     print(jl)
#     if jl<222:
#         # juli_list[str(j)] = x_list[i]
#         tmp_list.append(x_list[i])
#     else:
#         juli_list[str(j)] = tmp_list
#         tmp_list = []
#         j += 1
# if len(tmp_list)!=0:
#     juli_list[str(j)] = tmp_list
#     print('tmp',tmp_list)
#
#     # if len(np.argwhere(juli_list.keys()==j))==0 and len(juli_list)>0:
#     #     print(j)
#     #     juli_list[str(j)]==tmp_list
#     k += 1
#
# print(len(x_list))
# print(juli_list)
# val = list(juli_list.values())
# for i in val:
#     testli11111(i,img,(255,0,0))
ceshi = [[164, 158], [165, 138], [165, 155], [165, 156], [165, 157], [166, 138], [166, 157], [167, 137], [167, 138], [167, 139], [167, 140], [167, 157], [168, 137], [169, 136], [178, 117], [179, 120], [180, 118], [180, 119], [181, 117]]


y1_x = np.array(y5)[:,0]
y1_y = np.array(y5)[:,1]
y1_x_max = max(y1_x)
y1_y_max = max(y1_y)
y1_x_min = min(y1_x)
y1_y_min = min(y1_y)

a = np.argwhere(y1_x==y1_x_max)
b = np.argwhere(y1_x==y1_x_min)
c = np.argwhere(y1_y==y1_y_max)
d = np.argwhere(y1_y==y1_x_min)
# print(len(a))
# print(len(b))
# print(len(c))
# print(len(d))
ned_x = y5[int(c[0])][0]
ned_y = y5[int(c[0])][1]
# print(ned_x,ned_y)
kaishi = neighbours_x_y(ned_x,ned_y)
# print('start')
for i in kaishi:
    for j in ceshi:
        if i[0]==j[1] and i[1]==j[0]:
            pass
            # print(1)
            # print(i[0],i[1])
# testli11111(kaishi,img,(255,0,0))
# testli222222(ceshi,img,(255,0,0))

# 第一笔和fork点的交界处
y1_n = array([138,165])
# 第一笔和fork点的交界处
y2_n = array([158,164])
# y3_n = array([158,164])
# 第四笔和fork点的交界处1 右侧
y4_n_1 = array([155,165])
# 第四笔和fork点的交界处2 左侧
y4_n_2 = array([140,164])
# 第五笔和fork点的交界处1 左侧
y5_n_1 = array([120,174])
# 第五笔和fork点的交界处2 右侧
y5_n_2 = array([136,169])
t = np.array([233,17])
jl = compute_distance(t,y5_n_2)
# print(jl)

# testli11111([[156, 171]],img,(255,0,0))
# A,B,C,D,E, = [],[],[],[],[]
# {'bh0': [[138, 167], None],
# 'bh1': [[137, 168], [117, 181]],
# 'bh2': [[155, 165], [138, 167]],
# 'bh3': [(156, 171), None],
# 'bh4': [[117, 178], None]}

c= [[[138, 167], None], [[137, 168], [117, 181]], [[155, 165], [138, 167]], [(156, 171), None], [[117, 178], None]]
# for i in c:
#     for j in i:
#         print(j)
#         if j !=None:
            # testli11111([[j[0],j[1]]],img,(255,0,0))

# for i in range(4):
# testli11111(y1,img,(255,0,0))
# testli11111(y2,img,(255,255,0))
# testli11111(y3,img,(0,255,255))
# testli11111(y4,img,(0,255,0))
# testli11111(y5,img,(0,255,0))


# [[0. 1. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 1. 0. 0. 0. 1.]
#  [0. 0. 1. 0. 0. 1. 1. 1.]
#  [0. 0. 1. 1. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0. 0. 1. 0.]
#  [0. 0. 0. 1. 0. 1. 0. 0.]
#  [0. 0. 1. 1. 0. 0. 0. 0.]]
g_text = """
{
    0:[1,2],
    1:[2],
    2:[1,3,7],
    3:[2,5,6,7],
    4:[2,3,7],
    5:[3,6],
    6:[3,5],
    7:[2,3],

}
"""
import numpy as np
import math
import matplotlib.pyplot as plt

#
# def Translate(X,Y,angle,distance):                #defines function
#     # 0 degrees = North, 90 = East, 180 = South, 270 = West
#     dY = distance*math.cos\
#         (math.radians(angle))   #change in y
#     dX = distance*math.sin(math.radians(angle))   #change in x
#     Xfinal = X + dX
#     Yfinal = Y + dY
#     return Xfinal, Yfinal
#
#
# testli11111([[int(100), int(100)]], img, (255, 0, 0))
# juli_list = []
# for i in range(0,360,3):
#     for j in range(len(img)):
#          if j >100:
#              break
#          Xfinal, Yfinal = Translate(100,100,i,j)
#
#     testli11111([[int(Xfinal), int(Yfinal)]], img, (255, 0, 0))
# for i in x:
#     for j in y:

# 下午首先弄下距离公式
# 以距离公式得出波动图
# 测试波动图是否能完整获得点位信息
# 如 E  S  和 fork点
import numpy as np
# vector = [
#     0, 6, 25, 20, 15, 8, 15, 6, 0, 6, 0, -5, -15, -3, 4, 10, 8, 13, 8, 10, 3,
#     1, 20, 7, 3, 0 ]
vector =[144.5, 144.5, 182.5, 204.5, 188.5, 193.0, 198.5, 186.5, 212.5, 202.5, 212.0, 284.5, 216.5, 212.5, 226.0, 289.0, 272.5, 50.0, 37.0, 32.5, 32.5, 62.5, 36.5, 36.5, 54.5, 52.0, 52.0, 41.0, 41.0, 40.5, 40.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 6.5, 6.5, 6.5, 6.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# vector =[24.5, 24.5, 24.5, 25.0, 25.0, 42.5, 54.5, 54.5, 84.5, 102.5, 96.5, 104.0, 104.0, 101.0, 110.5, 144.0, 132.5, 145.0, 173.0, 162.5, 168.5, 185.0, 261.0, 274.0, 344.5, 389.0, 438.5, 590.5, 128.5, 72.0, 72.0, 61.0, 52.0, 52.0, 54.5, 54.5, 68.5, 102.5, 90.0, 96.5, 96.5, 112.5, 112.5, 156.5, 182.5, 210.5, 290.0, 293.0, 432.5, 1525.0, 1280.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 13.0, 13.0, 13.0, 14.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal #滤波等
from scipy.signal import medfilt
from scipy import arange
# vector = list(set(vector))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal  # 滤波等


ceshi = cv2.imread('images/img1.jpg')
print(ceshi.shape)
p_list =  [[174, 227], [174, 227], [176, 229], [177, 230], [178, 229], [179, 229], [180, 229], [181, 228], [182, 229], [183, 228], [184, 228], [187, 230], [186, 227], [187, 226], [188, 226], [191, 227], [191, 226], [182, 216], [181, 215], [181, 214], [181, 214], [184, 215], [182, 213], [182, 213], [184, 213], [184, 212], [184, 212], [183, 211], [183, 211], [183, 210], [183, 210], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [171, 208], [171, 208], [171, 208], [171, 208], [171, 209], [171, 209], [171, 209], [171, 209], [171, 209], [171, 209], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210]]

testli222222([[174, 210]],ceshi,(255,0,255))
# testli11111([[174, 211]],ceshi,(255,0,255))
# vector = list(map(int,vector))
# print(vector)
testli222222(p_list,ceshi,(255,0,255))



# xxx = np.arange(0, 120)
# # yyy = np.sin(xxx * np.pi / 180)
# yyy= vector
# z1 = np.polyfit(xxx, yyy, 50)  # 用7次多项式拟合
# p1 = np.poly1d(z1)  # 多项式系数
# print(p1)  # 在屏幕上打印拟合多项式
# yvals = p1(xxx)
# # 极值
# print(yvals[signal.argrelextrema(yvals, np.greater)]) #极大值的y轴, yvals为要求极值的序列
# print(signal.argrelextrema(yvals, np.greater)) #极大值的x轴
# peak_ind = signal.argrelextrema(yvals,np.greater)[0] #极大值点，改为np.less即可得到极小值点
# plt.plot(xxx, yyy, '*',label='original values')
# plt.plot(xxx, yvals, 'r',label='polyfit values')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.legend(loc=4)
# plt.title('polyfitting')
# plt.plot(signal.argrelextrema(yvals,np.greater)[0],yvals[signal.argrelextrema(yvals, np.greater)],'o', markersize=10) #极大值点
# plt.plot(signal.argrelextrema(yvals,np.less)[0],yvals[signal.argrelextrema(yvals, np.less)],'+', markersize=10) #极小值点
# plt.show()





















# vector = np.array(vector)*-1
# ceshi = medfilt(vector)
# print(ceshi)
#
# x = np.linspace(0,len(ceshi),len(ceshi))           # 设置x长度
# x_new = np.linspace(0,len(ceshi),len(ceshi))    # 10倍插值
# kind = "quadratic"                 # "quadratic","cubic" 为2阶、3阶B样条曲线插值
# f = interpolate.interp1d(x,x,kind=kind)
# DIS = f(x_new)
# plt.plot(x_new,ceshi) # 用插值后的绘制图像
# plt.xlabel('x')      # x轴标签
# plt.ylabel('y')      # y轴标签
# #plt.legend('')     # 标签
# plt.show()
# k_mean = np.mean(ceshi)
# num_peak_3 = signal.find_peaks(ceshi,height=k_mean) #distance表极大值点的距离至少大于等于10个水平单位
# # print(signal.argrelextrema(Dis)) #极大值的x轴
# print(ceshi)
# print(k_mean)
# print(num_peak_3[0])
# print(len(num_peak_3[0]))