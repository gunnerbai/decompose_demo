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

c =  [[174, 227], [174, 227], [176, 229], [177, 230], [178, 229], [179, 229], [180, 229], [181, 228], [182, 229], [183, 228], [184, 228], [187, 230], [186, 227], [187, 226], [188, 226], [191, 227], [191, 226], [182, 216], [181, 215], [181, 214], [181, 214], [184, 215], [182, 213], [182, 213], [184, 213], [184, 212], [184, 212], [183, 211], [183, 211], [183, 210], [183, 210], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [174, 209], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [173, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [172, 208], [171, 208], [171, 208], [171, 208], [171, 208], [171, 209], [171, 209], [171, 209], [171, 209], [171, 209], [171, 209], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210], [173, 210]]
vector =[144.5, 144.5, 182.5, 204.5, 188.5, 193.0, 198.5, 186.5, 212.5, 202.5, 212.0, 284.5, 216.5, 212.5, 226.0, 289.0, 272.5, 50.0, 37.0, 32.5, 32.5, 62.5, 36.5, 36.5, 54.5, 52.0, 52.0, 41.0, 41.0, 40.5, 40.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 6.5, 6.5, 6.5, 6.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# vector =[24.5, 24.5, 24.5, 25.0, 25.0, 42.5, 54.5, 54.5, 84.5, 102.5, 96.5, 104.0, 104.0, 101.0, 110.5, 144.0, 132.5, 145.0, 173.0, 162.5, 168.5, 185.0, 261.0, 274.0, 344.5, 389.0, 438.5, 590.5, 128.5, 72.0, 72.0, 61.0, 52.0, 52.0, 54.5, 54.5, 68.5, 102.5, 90.0, 96.5, 96.5, 112.5, 112.5, 156.5, 182.5, 210.5, 290.0, 293.0, 432.5, 1525.0, 1280.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 13.0, 13.0, 13.0, 14.5, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# vector =[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.5, 4.5, 4.5, 4.5, 4.5, 25.0, 25.0, 132.5, 474.5, 490.5, 572.5, 522.0, 624.5, 740.0, 868.5, 1232.5, 1557.0, 2041.0, 552.5, 380.5, 344.5, 312.5, 85.0, 62.5, 62.5, 58.0, 58.0, 65.0, 65.0, 52.0, 41.0, 41.0, 32.0, 32.0, 32.0, 32.5, 32.5, 26.5, 26.5, 26.5, 29.0, 29.0, 29.0, 32.5, 32.5, 44.5, 50.0, 42.5, 36.0, 36.0, 36.0, 56.5, 65.0, 92.5, 349.0, 302.5, 292.0, 282.5, 320.0, 362.5, 277.0, 272.5, 292.5, 266.5, 220.5, 221.0, 202.0, 204.5, 185.0, 212.5, 92.5, 20.0, 14.5, 14.5, 10.0, 10.0, 10.0, 10.0, 26.0, 30.5, 25.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
# vector =[8.0, 8.0, 8.0, 8.0, 8.5, 8.5, 8.5, 8.5, 20.0, 29.0, 53.0, 53.0, 65.0, 65.0, 72.5, 81.0, 72.5, 82.0, 104.0, 104.0, 116.5, 137.0, 169.0, 146.0, 218.0, 254.5, 325.0, 320.5, 425.0, 762.5, 1512.5, 1356.5, 130.0, 76.5, 76.5, 68.5, 68.5, 84.5, 62.5, 68.0, 68.0, 112.5, 112.5, 132.5, 144.0, 306.5, 197.0, 260.0, 354.5, 509.0, 1592.5, 1254.5, 146.0, 65.0, 65.0, 52.0, 41.0, 12.5, 12.5, 12.5, 12.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# vector =[18.0, 18.0, 18.0, 18.5, 18.5, 18.5, 34.0, 45.0, 73.0, 73.0, 130.0, 85.0, 92.5, 101.0, 90.5, 100.0, 132.5, 122.0, 188.5, 153.0, 153.0, 160.0, 194.0, 252.5, 266.5, 330.5, 438.5, 524.5, 768.5, 128.0, 128.0, 72.5, 62.5, 62.5, 54.5, 54.5, 58.0, 58.0, 109.0, 96.5, 96.5, 112.5, 112.5, 145.0, 169.0, 420.5, 226.0, 312.5, 432.5, 1681.0, 1325.0, 116.0, 45.0, 34.0, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 6.5, 6.5, 6.5, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# vector =[144.5, 144.5, 182.5, 204.5, 188.5, 193.0, 198.5, 186.5, 212.5, 202.5, 212.0, 284.5, 216.5, 212.5, 226.0, 289.0, 272.5, 50.0, 37.0, 32.5, 32.5, 62.5, 36.5, 36.5, 54.5, 52.0, 52.0, 41.0, 41.0, 40.5, 40.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 6.5, 6.5, 6.5, 6.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

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



import numpy as np

#!/usr/bin/env python
# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
import numpy as np
import pylab


import  scipy.signal as signal

import math

class Point:
    """
    2D坐标点
    """
    def __init__(self, x, y):
        self.X = x
        self.Y = y


class Line:
    def __init__(self, point1, point2):
        """
        初始化包含两个端点
        :param point1:
        :param point2:
        """
        self.Point1 = point1
        self.Point2 = point2


def GetAngle(line1, line2):
    """
    计算两条线段之间的夹角
    :param line1:
    :param line2:
    :return:
    """
    dx1 = line1.Point1.X - line1.Point2.X

    dy1 = line1.Point1.Y - line1.Point2.Y
    dx2 = line2.Point1.X - line2.Point2.X
    dy2 = line2.Point1.Y - line2.Point2.Y
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        insideAngle = abs(angle1 - angle2)
    else:
        insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle = insideAngle % 180
    return insideAngle


def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle



# class Point(object):
#     def __init__(self, x=0, y=0):
#         self.x = x
#         self.y = y
#
#
# class Line(object):  # 直线由两个点组成
#     def __init__(self, p1=Point(0, 0), p2=Point(2, 2)):
#         self.p1 = p1
#         self.p2 = p2
#
#     def distance_point_to_line(self, current_line, mainline):
#         angle = self.get_cross_angle(current_line, mainline)
#         sin_value = np.sin(angle * np.pi / 180)  # 其中current_line视为斜边
#         long_edge = math.sqrt(  # 获取斜边长度
#             math.pow(current_line.p2.x - current_line.p1.x, 2) + math.pow(current_line.p2.y - current_line.p1.y,
#                                                                           2))  # 斜边长度
#         distance = long_edge * sin_value
#         return distance
#
#     def get_cross_angle(self, l1, l2):
#         arr_a = np.array([(l1.p2.x - l1.p1.x), (l1.p2.y - l1.p1.y)])  # 向量a
#         arr_b = np.array([(l2.p2.x - l2.p1.x), (l2.p2.y - l2.p1.y)])  # 向量b
#         cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  # 注意转成浮点数运算
#         return np.arccos(cos_value) * (180 / np.pi)  # 两个向量的夹角的角度， 余弦值：cos_value, np.cos(para), 其中para是弧度，不是角度
#
#     def get_main_line(self, mask):
#         # 获取最上面和最下面的contour的质心
#         contour_list = get_contours(mask)
#         c7_x, c7_y = get_centroid(contour_list[7])  # 最上面
#         c0_x, c0_y = get_centroid(contour_list[0])  # 最下面
#
#         # 获取串联两个质心，得到主线
#         point1 = Point(c7_x, c7_y)
#         point2 = Point(c0_x, c0_y)
#         mainline = Line(point1, point2)
#         return mainline
#
#     #  求seg_img图中的直线与垂直方向的夹角
#     def mainline_inclination_angle(self, seg_img):
#         # 获取串联两个质心，得到主线
#         mainline = self.get_main_line(seg_img)
#         # 测试该函数，三角形边长：3,4,5
#         mainline.p1.x = 0  # 列
#         mainline.p1.y = 0
#         mainline.p2.x = 3
#         mainline.p2.y = 4
#         # 获取参考线，这里用的是垂直方向的直线
#         # base_line = Line.get_main_line(normal_mask)
#         base_line = Line(Point(mainline.p1.x, mainline.p1.y),
#                          Point(mainline.p1.x, mainline.p2.y))  # 同一列mainline.p1.x，行数随便
#         # 获取两条线的夹角
#         angle = mainline.get_cross_angle(mainline, base_line)
#         return angle


# 求最大连通域的中心点坐标
def get_centroid(contour):
    moment = cv2.moments(contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy  # col, row
    else:
        return None


# 过滤面积较小的contour, 不同于后面的get_cnts(),
def get_contours(seg_img, area_thresh=0):
    if len(seg_img.shape) == 3:
        image_gray = cv2.cvtColor(seg_img, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = seg_img.copy()

    # 1, 二值化原图的灰度图,然后求解轮廓contours
    _, mask_contour = cv2.threshold(image_gray.copy(), 0.1, 255, cv2.THRESH_BINARY)
    # 2, 找二值化后图像mask中的contour
    contours, hierarchy = cv2.findContours(image=mask_contour.copy(), mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
    # 3, 遍历轮廓,将轮廓面积大于阈值的保存到contour_list中
    contour_list = []
    if len(contours) == 0:
        return contour_list
    for c_num in range(len(contours)):
        area = cv2.contourArea(contours[c_num])
        if area > area_thresh:
            contour_list.append(contours[c_num])
        else:
            continue
    return contour_list


def get_bbox_num(seg_img, area_thresh=0):
    image = seg_img.copy()
    contour_list = get_contours(image, area_thresh)  # contours排序是从上到下
    return len(contour_list)

bihua_lines = [[(216, 176), (227, 176)],
 [(189, 178), (210, 178)],
 [(222, 179), (212, 209)],
 [(211, 208), (184, 193)],
 [(212, 210), (287, 248)],
 [(190, 182), (184, 192)],
 [(179, 196), (139, 234)],
 [(210, 216), (177, 241)]]

# !usr/bin/env python
# encoding:utf-8


'''
__Author__:沂水寒城
功能：  相似度度量准则总结实现
'''

import math
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

import sys





def pearsonrSim(x, y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x, y)[0]


def spearmanrSim(x, y):
    '''
    斯皮尔曼相似度
    '''
    return spearmanr(x, y)[0]


def kendalltauSim(x, y):
    '''
    肯德尔相似度
    '''
    return kendalltau(x, y)[0]


def cosSim(x, y):
    '''
    余弦相似度计算方法
    '''
    tmp = sum(a * b for a, b in zip(x, y))
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return round(tmp / float(non), 3)


def eculidDisSim(x, y):
    '''
    欧几里得相似度计算方法
    '''
    return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattanDisSim(x, y):
    '''
    曼哈顿距离计算方法
    '''
    return sum(abs(a - b) for a, b in zip(x, y))


def minkowskiDisSim(x, y, p):
    '''
    明可夫斯基距离计算方法
    '''
    sumvalue = sum(pow(abs(a - b), p) for a, b in zip(x, y))
    tmp = 1 / float(p)
    return round(sumvalue ** tmp, 3)




def jaccardDisSim(x, y):
    '''
    杰卡德相似度计算
    '''
    res = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return res / float(union_cardinality)


if __name__ == '__main__':
    # x = [1, 2, 3, 4, 5]
    # y = [2, 4, 0, 8, 9]
    fork_list = [177, 223], [178, 222], [179, 222], [180, 190], [181, 190], [182, 190], [182, 191], [182, 192], [183, 189], [191,
                                                                                                                 185], [
        192, 184], [193, 184], [194, 183], [208, 213], [209, 212], [210, 212], [213, 212], [214, 212], [215, 211], [215,
                                                                                                                    212], [
        215, 213], [216, 210]

    # for i in range(len(fork_list)):
    #     for j in range(len(fork_list)):
    #         if i !=j:
    #             img = cv2.imread('images/test2.png')
    #             x = fork_list[i]
    #             y = fork_list[j]
    #
    #             # print('pearsonrSim:', pearsonrSim(x, y))
    #             # print( 'spearmanrSim:', spearmanrSim(x, y))
    #             # print( 'kendalltauSim:', kendalltauSim(x, y))
    #             # print('cosSim:', cosSim(x, y))
    #             # print('eculidDisSim:', eculidDisSim(x, y))
    #             print( 'manhattanDisSim:', manhattanDisSim(x, y))
    #             # print( 'minkowskiDisSim:', minkowskiDisSim(x, y, 2))
    #             # print('jaccardDisSim:', jaccardDisSim(x, y))
    #             print('-------------------------------------------------')
    #             testli11111([x], img, (255, 0, 0))
    #             testli11111([y], img, (255, 0, 0))
    # 0 0 1
    # p02p1 = [(216, 176),(221, 179),(210, 178)]
    # 1 0 2
    # p02p1 = [(189, 178),(221, 179),(212, 209)]
    # 0
    # 0
    # 2
    # p02p1 = [(216, 176), [221, 179], (212, 209)]
    # 1
    # 1
    # 5
    p02p1 = [(189, 178), [191, 181], (184, 192)]
    # 3
    # 2
    # 5
    # p02p1 = [(211, 208), [183, 193], (184, 192)]
    # 3
    # 2
    # 6
    # p02p1 = [(211, 208), [183, 193], (139, 234)]
    # 5
    # 2
    # 6
    # p02p1 = [(190, 182), [183, 193], (139, 234)]
    # 2
    # 3
    # 3
    # p02p1 = [(222, 179), [211, 208], (184, 193)]
    # 2
    # 3
    # 4
    # p02p1 = [(222, 179), [211, 208], (287, 248)]
    # 2
    # 3
    # 7
    # p02p1 = [(222, 179), [211, 208], (177, 241)]
    # 3
    # 3
    # 4
    # p02p1 = [(211, 208), [211, 208], (287, 248)]
    # 3
    # 3
    # 7
    # p02p1 = [(211, 208), [211, 208], (177, 241)]
    # 4
    # 3
    # 7
    # p02p1 = [(212, 210), [211, 208], (177, 241)]
    # test_iamge = cv2.imread('images/test2.png')
    # testli11111([(184, 193), [211, 208], (287, 248)],test_iamge,(255,0,0))
    import numpy as np
    import math


    def latlong_to_3d(latr, lonr):
        """Convert a point given latitude and longitude in radians to
        3-dimensional space, assuming a sphere radius of one."""
        return np.array((
            math.cos(latr) * math.cos(lonr),
            math.cos(latr) * math.sin(lonr),
            math.sin(latr)
        ))


    def angle_between_vectors_degrees(u, v):
        """Return the angle between two vectors in any dimension space,
        in degrees."""
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


    # The points in tuple latitude/longitude degrees space
    # A = (12.92473, 77.6183)
    # B = (12.92512, 77.61923)
    # C = (12.92541, 77.61985)
    A = p02p1[0]
    B = p02p1[1]
    C = p02p1[2]
    # Convert the points to numpy latitude/longitude radians space
    a = np.radians(np.array(A))
    b = np.radians(np.array(B))
    c = np.radians(np.array(C))

    # Vectors in latitude/longitude space
    avec = a - b
    cvec = c - b

    # Adjust vectors for changed longitude scale at given latitude into 2D space
    lat = b[0]
    avec[1] *= math.cos(lat)
    cvec[1] *= math.cos(lat)

    # Find the angle between the vectors in 2D space
    angle2deg = angle_between_vectors_degrees(avec, cvec)

    # The points in 3D space
    a3 = latlong_to_3d(*a)
    b3 = latlong_to_3d(*b)
    c3 = latlong_to_3d(*c)

    # Vectors in 3D space
    a3vec = a3 - b3
    c3vec = c3 - b3

    # Find the angle between the vectors in 2D space
    angle3deg = angle_between_vectors_degrees(a3vec, c3vec)

    # Print the results
    print('\nThe angle ABC in 2D space in degrees:', angle2deg)
    print('\nThe angle ABC in 3D space in degrees:', angle3deg)