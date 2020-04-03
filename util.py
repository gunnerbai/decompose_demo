import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from libs.tony_beltramelli_detect_peaks import detect_peaks
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
    DIS = f(x_new)
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



# q = [[1., 1., 1., 0., 0., 0., 0., 0.],
#     [0. ,1., 0., 0., 0., 1., 0., 0.],
#     [0., 0., 0., 1., 0., 1., 1., 0.],
#     [0., 0., 1., 1., 1., 0., 0., 1.]]
#
#
# if __name__ == '__main__':
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