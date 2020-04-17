import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from libs.tony_beltramelli_detect_peaks import detect_peaks
import copy
fork_list = [[177, 223], [178, 222], [179, 222], [180, 190], [181, 190], [182, 190], [182, 191], [182, 192], [183,189],\
            [191,185], [192, 184], [193, 184], [194, 183], [208, 213], [209, 212], [210, 212], [213, 212], [214, 212],\
            [215,211], [215,212], [215, 213], [216, 210]]
bihua_lines =  [[(216, 176), (227, 176)], [(189, 178), (210, 178)], [(222, 179), (212, 209)], [(211, 208), (184, 193)], [(212, 210), (287, 248)], [(190, 182), (184, 192)], [(179, 196), (139, 234)], [(210, 216), (177, 241)]]
tmp = [[221, 179], [221, 177], [223, 178], [189, 181], [191, 181], [191, 182], [190, 183], [189, 182], [191, 181], [191, 183], [190, 181], [191, 181], [183, 193], [185, 192], [185, 194], [211, 210], [211, 208], [213, 209], [211, 214], [213, 213], [213, 214], [213, 215], [211, 216], [210, 216], [212, 214], [212, 216], [211, 215], [212, 214], [212, 214], [213, 214], [214, 216]]

def manhattanDisSim(x, y):
    '''
    曼哈顿距离计算方法
    '''
    return sum(abs(a - b) for a, b in zip(x, y))

def testli11111(oth,image,ys):
    for n in range(len(oth)) :
        if oth[n]!=None:
        # print('oth:',oth[n])
            cv2.circle(image, (int(oth[n][0]),int(oth[n][1])), 1 , ys )
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1000, 1000)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    return image

# fork点分类
def getfork(fork_list):

    fork = {}
    fork_list = list(fork_list)
    fork_list1 = fork_list.copy()
    nu = 0
    for i in fork_list:
        cunchu_list = []
        cunchu_list.append(i)
        fork_list.remove(i)
        for j in fork_list1:
            if i !=j:
                x = i
                y = j
                # print(x,y)
                # print('manhattanDisSim:', manhattanDisSim(x, y))
                # print('-------------------------------------------------------')
                if  manhattanDisSim(x, y)<15:
                        cunchu_list.append(y)
                        fork_list.remove(y)
        fork[str(nu)]=cunchu_list
        nu+=1
    # print(cunchu)
    return fork

# 获得笔画交叉点
def get_stroke_intersection(fork_list,all_stroke):
    jiaolian = []
    for i in range(len(all_stroke)):
        # print(all_stroke[i])
        p1 = all_stroke[i][0]
        p2 = all_stroke[i][1]
        keys = fork_list.keys()
        for j in keys:
            z1 = manhattanDisSim(p1, fork_list.get(j)[0])
            z2 = manhattanDisSim(p2, fork_list.get(j)[0])
            if z1<15 :
                jiaolian.append([i,j])
            elif z2<15:
                jiaolian.append([i, j])
    return jiaolian

def  get_PBOD(point,image,line):
    BF_Pision_list = []
    Dis = []
    jd = []
    p1 = line[0]
    p2 = line[1]
    jl1 = compute_distance(p1,point)
    jl2 = compute_distance(p2,point)
    if  jl1<jl2:
        p = p2
    else:
        p =p1
    # testli11111([p], image, (255, 0, 0))

    for i in range(0, 360):
        for j in range(len(image)):
            if i ==299:
                print(1)
            # 根据点位 角度 距离  获得 新的射线x y
            sx_x, sx_y = Translate(point[0], point[1], i, j)
            # print(1)
            # testli11111([[sx_x,sx_y]], image, (255, 0, 0))
            sx_x_u = int(sx_x)  # 向下取整
            sx_x_d = math.ceil(sx_x)  # 向上取整
            sx_y_u = int(sx_y)  # 向下取整
            sx_y_d = math.ceil(sx_y)  # 向上取整
            if (int(sx_x_u), int(sx_y_u))==p or (int(sx_x_u), int(sx_y_d))==p or (int(sx_x_d), int(sx_y_d))==p or (int(sx_x_d), int(sx_y_u))==p:
                # print('this is true')
                # print(i)
                jd.append(i)
                # print((int(sx_y), int(sx_x)))
            # print([int(sx_y), int(sx_x)])
            # testli11111([[int(sx_y), int(sx_x)]],image,(255,0,0))
            if image[int(sx_y), int(sx_x)] == 0:
                # testli11111([point, (sx_y, sx_x)], img, (255, 0, 0))
                BF_Pision_list.append([int(sx_x), int(sx_y)])
                # 计算距离
                jl = compute_distance((point[0], point[1]), (int(sx_x), int(sx_y)))
                Dis.append(jl)
                break
    if len(jd)!=0:
        return Dis, BF_Pision_list, min(jd)
    return Dis, BF_Pision_list

def  get_PBOD2(point,image,poing2):
    image =255-image
    BF_Pision_list = []
    Dis = []
    jd = []
    p = poing2
    for i in range(0, 360):
        for j in range(len(image)):
            # 根据点位 角度 距离  获得 新的射线x y
            sx_x, sx_y = Translate(point[0], point[1], i, j)
            # testli11111([[sx_x,sx_y]], image, (255, 0, 0))
            sx_x_u = int(sx_x)  # 向下取整
            sx_x_d = math.ceil(sx_x)  # 向上取整
            sx_y_u = int(sx_y)  # 向下取整
            sx_y_d = math.ceil(sx_y)  # 向上取整
            if (int(sx_x_u), int(sx_y_u))==p or (int(sx_x_u), int(sx_y_d))==p or (int(sx_x_d), int(sx_y_d))==p or (int(sx_x_d), int(sx_y_u))==p:
                jd.append(i)
            if image[int(sx_y), int(sx_x)] == 0:
                # testli11111([point, (sx_y, sx_x)], img, (255, 0, 0))
                BF_Pision_list.append([int(sx_x), int(sx_y)])
                # 计算距离
                jl = compute_distance((point[0], point[1]), (int(sx_x), int(sx_y)))
                Dis.append(jl)
                break
    if len(jd)!=0:
        return Dis, BF_Pision_list, min(jd)
    return Dis, BF_Pision_list


#all_strokes  为所有笔画
#多少位置需要确定笔画权重的数组
def get_strokes_Weight(all_strokes,index ):
    weight_list = []
    # print('122222',index)
    for i in range(len(all_strokes)):
        if index ==None:
                    A_x = 0
                    A_y = 0
                    point_list = np.array(all_strokes[i].points)
                    x_list = point_list[:, 0]
                    y_list = point_list[:, 1]
                    point_max_x = max(x_list)
                    point_max_y = max(y_list)
                    point_min_x = min(x_list)
                    point_min_y = min(y_list)
                    kd = (point_max_x + 1) - point_min_x
                    gd = (point_max_y + 1) - point_min_y
                    for k in range(point_min_x, point_max_x + 1):
                        A_x += k

                    for j in range(point_min_y, point_max_y + 1):
                        A_y += j
                    A_x = A_x / (kd)
                    A_y = A_y / (gd)
                    weight_list.append([int(A_x), int(A_y)])
        else:
            for j in range (len(index)):
                if i ==index[j]:
                    A_x = 0
                    A_y = 0
                    point_list = np.array(all_strokes[i].points)
                    x_list = point_list[:, 0]
                    y_list = point_list[:, 1]
                    point_max_x = max(x_list)
                    point_max_y = max(y_list)
                    point_min_x = min(x_list)
                    point_min_y = min(y_list)
                    kd = (point_max_x + 1) - point_min_x
                    gd = (point_max_y + 1) - point_min_y
                    for k in range(point_min_x, point_max_x + 1):
                        A_x += k

                    for j in range(point_min_y, point_max_y + 1):
                        A_y += j
                    A_x = A_x / (kd)
                    A_y = A_y / (gd)
                    weight_list.append([int(A_x), int(A_y)])

    return weight_list


def  get_PBOD3(point,image,line):
    image = cv2.bitwise_not(image)
    testimage = cv2.imread('testline/58117.png')
    jd = []
    p1 = line[0]
    p2 = line[1]
    jl1 = compute_distance(p1,point)
    jl2 = compute_distance(p2,point)
    if  jl1<jl2:
        p = p2
    else:
        p =p1
    # testli11111([p], testimage, (255, 0, 0))
    # testli11111([point], testimage, (255, 0, 0))
    for i in range(0, 360):

        for j in range(len(image)):
            # 根据点位 角度 距离  获得 新的射线x y
            sx_x, sx_y = Translate(point[1], point[0], i, j)

            sx_x_u = int(sx_x)  # 向下取整
            sx_x_d = math.ceil(sx_x)  # 向上取整
            sx_y_u = int(sx_y)  # 向下取整
            sx_y_d = math.ceil(sx_y)  # 向上取整
            # cv2.circle(testimage, (int(sx_y), int(sx_x)), 1, (255, 255, 0))
            if (int(sx_x_u), int(sx_y_u))==p or (int(sx_x_u), int(sx_y_d))==p or (int(sx_x_d), int(sx_y_d))==p or (int(sx_x_d), int(sx_y_u))==p:
                # print(i)

                jd.append(i)
            if image[int(sx_x), int(sx_y)] == 0:
                break
    # testli11111([[sx_y, sx_x]], testimage, (255, 0, 0))
    if len(jd) == 0:
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return 5000

    else:
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return min(jd)

def  get_PBOD4(point,image,line):
    image =255-image
    jd = []
    Dis = []
    p = line
    testimage = cv2.imread('testline/58117.png')
    # testli11111([p], testimage, (255, 0, 0))
    for i in range(0, 360):
        for j in range(len(image)):
            sx_x, sx_y = Translate(point[1], point[0], i, j)
            sx_x_u = int(sx_x)  # 向下取整
            sx_x_d = math.ceil(sx_x)  # 向上取整
            sx_y_u = int(sx_y)  # 向下取整
            sx_y_d = math.ceil(sx_y)  # 向上取整
            # cv2.circle(testimage, (int(sx_y), int(sx_x)), 1, (255, 255, 0))
            print('ssssssssssss',sx_y,sx_x)
            print('dddddddddddd',p)
            # testli11111([[sx_y, sx_x]], testimage, (255, 255, 0))
            if (int(sx_y_u), int(sx_x_u))==p or (int(sx_y_u), int(sx_x_d))==p or (int(sx_y_d), int(sx_x_d))==p or (int(sx_y_d), int(sx_x_u))==p:
                print('pbod__4',i)
                # testli11111([[sx_y, sx_x]], testimage, (255, 255, 0))
                jd.append(i)
            if image[int(sx_x), int(sx_y)] == 0:
                # 计算距离
                jl = compute_distance((point[0], point[1]), (int(sx_x), int(sx_y)))
                Dis.append(jl)
                break
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # testli11111([[sx_y, sx_x]], testimage, (255, 0, 0))
    if len(jd)==0:
        show_image(Dis)
        return 5000
    else:
        show_image(Dis)
        return min(jd)


def Translate(X,Y,angle,distance):                #defines function
    # 0 degrees = North, 90 = East, 180 = South, 270 = West
    dY = distance*math.cos\
        (math.radians(angle))   #change in y
    dX = distance*math.sin(math.radians(angle))   #change in x
    Xfinal = X + dX
    Yfinal = Y + dY
    return Xfinal, Yfinal


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

def get_angel(line1,line2,fork):
    import angel
    jl1 = manhattanDisSim(line1[0],fork)
    jl2 = manhattanDisSim(line1[1],fork)
    jl3 = manhattanDisSim(line2[0],fork)
    jl4 = manhattanDisSim(line2[1],fork)
    if jl1>15:
        an1 = angel.get_this_angel(line1[0],fork,line2[0])
    else:
        an1 = [0]

    if jl2 > 15:
        an2 = angel.get_this_angel(line1[1],fork,line2[1])
    else:
        an2 = [0]

    if jl3 > 15:
        an3 = angel.get_this_angel(line1[1],fork,line2[0])
    else:
        an3 = [0]

    if jl4 > 15:
        an4 = angel.get_this_angel(line1[0],fork,line2[1])
    else:
        an4 = [0]

    print('an1:',an1[0])
    print('an2:',an2[0])
    print('an3:',an3[0])
    print('an4:',an4[0])


def show_image(Dis):
    x = np.linspace(0, len(Dis), len(Dis))  # 设置x长度
    x_new = np.linspace(0, len(Dis), len(Dis))  # 10倍插值
    kind = "quadratic"  # "quadratic","cubic" 为2阶、3阶B样条曲线插值
    f = interpolate.interp1d(x, x, kind=kind)
    plt.plot(x_new, Dis)  # 用插值后的绘制图像
    plt.xlabel('x')  # x轴标签
    plt.ylabel('y')  # y轴标签
    # plt.legend('')     # 标签
    plt.show()


def get_peak(Dis):
    indexes = detect_peaks(Dis, 0.6)
    # print(indexes)
    quchu_list = []
    for index in range(len(indexes) - 1):
        if abs(indexes[index] - indexes[index + 1]) <= 10:
            quchu_list.append(indexes[index])
    indexes = list(indexes)
    # print(quchu_list)
    for quchu in quchu_list:
        indexes.remove(quchu)
    print(indexes)
    print('indexes:', indexes)
    print('峰值是：', len(indexes))
    return indexes
def  remove_this_list(remove_list):
    for index1,i in enumerate(remove_list):
        for index2,j in enumerate(remove_list):
            if index1 != index2 :
                if i[0]==j[1] and  i[1]==j[0]:
                    remove_list.remove(j)
                    return remove_list
                elif i[0]==j[0] and i[1]==j[1]:
                    remove_list.remove(j)
                    return remove_list
    return  0


def  remove_list_repetition(re_list):
    testlist= copy.deepcopy(re_list)
    j =0
    while True:
        dest = remove_this_list(testlist)
        if dest==0:
            # testlist =dest[0]
            return testlist
        else:
            testlist =dest
        j+=1

def repetition(source):
    dest = []
    for e in source:
        if e not in dest:
            dest.append(e)
    return dest
def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    # if  x=10.y=10   3点钟方向 逆时针 点位
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    # testli11111([[x,y1] ,[x_1,y1], [x_1,y],  [x_1,y_1], [x,y_1] ,[x1,y_1], [x1,y], [x1,y1]],img,(255,0,0))
                 # p0        p1            p2               p3          p4              p5          p6             p7
    return [img[y1][x] ,img[y1][x_1], img[y][x_1],  img[y_1][x_1], img[y_1][x] , img[y_1][x1],  img[y][x1], img[y1][x1]]

def Independent_stroke(stroke_i,skimage):
    pdlist = []
    # 找到头部和尾部
    p1 = stroke_i.starting_point
    p2 = stroke_i.ending_point
    # 判断头部或者尾部是否有链接
    n1 = neighbours(p1[0],p1[1],skimage)
    n2 = neighbours(p2[0],p2[1],skimage)
    # print(n1,n2)
    n1 = np.argwhere(np.array(n1)==255)
    n2 = np.argwhere(np.array(n2)==255)
    if len(n1)==1:
        # print('{}笔画的起始点是 独立的'.format(index))
        pdlist.append(True)
    else:
        pdlist.append(False)
    if len(n2)==1:
        # print('{}笔画的结束点是 独立的'.format(index))
        pdlist.append(True)
    else:
        pdlist.append(False)
    return pdlist

def distance(pixel_1, pixel_2):
    delta_x = (pixel_1[0] - pixel_2[0])**2
    delta_y = (pixel_1[1] - pixel_2[1]) ** 2
    return (delta_x + delta_y)**0.5

def  get_jiaodian(test_list):
        dd1 = []
        dd2 = []
        for i in range(len(test_list) - 1):
            jl = distance(test_list[i][-1], test_list[i + 1][-1])
            if jl < 5:
                q = []
                q.extend(test_list[i][0:-1])
                q.extend(test_list[i + 1][0:-1])
                q.append(test_list[i][-1])
                dd1.append(q)
            else:
                dd2.append(test_list[i + 1])
        return dd1, dd2


def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """

    a = math.sqrt((point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt((point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt((point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
    return A,B,C

# q = [[1., 1., 1., 0., 0., 0., 0., 0.],
#     [0. ,1., 0., 0., 0., 1., 0., 0.],
#     [0., 0., 0., 1., 0., 1., 1., 0.],
#     [0., 0., 1., 1., 1., 0., 0., 1.]]
#
#
if __name__ == '__main__':
    s = cal_ang([0,0],[-10,2],[10,0])
    print(s)
#     fork  = getfork(tmp)
#     # get_angel(bihua_lines[0], bihua_lines[1], fork.get('0')[0])
#     print(fork.get('3')[0])
#     # print(fork)
#     # for i in fork.keys():
#     # img = cv2.imread('images/test2.png')
#     # testli11111([bihua_lines[1][0],fork.get('0')[0],bihua_lines[2][0]], img, (255, 0, 0))
#     # jiaolian= get_stroke_intersection(fork,bihua_lines)







    # print(jiaolian)
    # for i in jiaolian:
        # get_angel(bihua_lines[i[]])
        # img = cv2.imread('images/test2.png')
        # print(i)
        # testli11111(bihua_lines[i[0]],img,(255,0,0))
        # testli11111(fork.get(i[1]),img,(255,0,255))