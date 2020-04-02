import random
import math
import  os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_ubyte


#key: row and column of the neighbouring pixels, value: the amount to be added to the central pixel -> h,wpl.,
delta = {(0,0): (-1,-1), (0,1): (-1,0), (0,2): (-1,1),
            (1,0): (0,-1), (1,1): (0,0), (1,2): (0,1),
            (2, 0): (1, -1), (2,1): (1,0), (2,2): (1,1)}

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (80, 127, 255), (255, 0, 255),
          (255, 255, 0), (96, 164, 244)]

def testli(oth,image,ys):
    i = image.copy()
    for n in range(len(oth)) :
        # print('oth:',oth[n])
        cv2.circle(i, (oth[n][0],oth[n][1]), 1 , ys )
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img', 1000, 1000)
    # cv2.imshow('img',image)
    # cv2.waitKey(0)
    return i

def testli11111(oth,image,ys):
    for n in range(len(oth)) :
        # if oth[n]!=None:
        # print('oth:',oth[n])
            cv2.circle(image, (oth[n][0],oth[n][1]), 1 , ys )
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 1000, 1000)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    return image

class Stroke(object):
    class_counter= 0

    def __init__(self, points, image):

        self.points = points
        self.image = image
        self.starting_point, self.ending_point = self.find_ending_points()
        self.index = Stroke.class_counter
        Stroke.class_counter += 1

    def draw_strokes3(self, image, points):
        if len(image.shape) >2:
            h, w, d = image.shape
        else :
            h, w= image.shape
        blank_image = np.zeros((h, w), np.uint8)

        for p in points:
            blank_image[p[1], p[0]] = 255

        return blank_image

    def distance(self, pixel_1, pixel_2):
        delta_x = (pixel_1[0] - pixel_2[0]) ** 2
        delta_y = (pixel_1[1] - pixel_2[1]) ** 2
        return (delta_x + delta_y) ** 0.5

    def far_away_end_points(self, ending_points):
        max_dist = -10
        ending_points_ = None
        for p in ending_points:
            for p2 in ending_points:
                d = self.distance(p, p2)
                if d > max_dist:
                    ending_points_ = [p, p2]
                    max_dist = d
        return ending_points_



    def compareN(self, neighbourhood):
        possible = [np.array([[0, 255, 255], [0, 255, 0], [0, 0, 0]]),
                    np.array([[255, 255, 0], [0, 255, 0], [0, 0, 0]]),
                    np.array([[0, 0, 0], [0, 255, 0], [255, 255, 0]]),
                    np.array([[0, 0, 0], [0, 255, 0], [0, 255, 255]]),
                    np.array([[0, 0, 255], [0, 255, 255], [0, 0, 0]]),
                    np.array([[255, 0, 0], [255, 255, 0], [0, 0, 0]]),
                    np.array([[0, 0, 0], [255, 255, 0], [255, 0, 0]]),
                    np.array([[0, 0, 0], [0, 255, 255], [0, 0, 255]])
                    ]
        for p in possible:

            c = neighbourhood == p
            if c.all():
                return True
        return False

    def find_ending_points(self):
        d = self.draw_strokes3(self.image, self.points)
        # cv2.imshow('1',d)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ending_points = []
        for p in self.points:
            neighbourhood = list()
            neighbourhood[:] = d[p[1] - 1: p[1] + 2, p[0] - 1: p[0] + 2]
            neighbours = np.argwhere(neighbourhood)
            print('neighbours', neighbours, 'len neighbours', len(neighbours))
            if len(neighbours) <= 2:
                ending_points.append(p)
            elif len(neighbours) == 3:
                if  self.compareN(neighbourhood):
                    ending_points.append(p)
        # returns the two ending points that are more far away
        print("points", self.points)
        print("ending_points", ending_points)
        # if no ending points were found we have a close stroke (i.e. a perfectly closed circle)
        if not ending_points:
            return (self.points[0], self.points[1])

        # testli(ending_points,ima,(255,255,0))

        real_ending_points = self.far_away_end_points(ending_points)

        # testli([real_ending_points[0], real_ending_points[-1]], ima, (255, 255, 0))

        return (real_ending_points[0], real_ending_points[-1])


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
    # cv2.imshow('111',threshold)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    threshold[threshold == 255] = 1
    # ceshi.append(threshold[threshold == 1])
    skeleton = skeletonize(threshold)
    skeleton = img_as_ubyte(skeleton)
    h , w = skeleton.shape
    for i in range(h):
        for j in range(w):
            if skeleton[i,j]==255:
                ceshi.append([i,j])
    return skeleton,ceshi

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

def find_top_left_most_pixel(skeleton_image, processing_index = 0):
    """
    Expects an skeletonized image (binary image with one-pixel width lines)
    """
    for y in range(processing_index, skeleton_image.shape[0], 1):
        for x in range(0, skeleton_image.shape[1]):

            if skeleton_image[y, x] == 255 :
                return (x,y)
    return None

def inside_image(pixel, image):
    """Checks whether a pixel is inside the image space"""
    h, w = image.shape
    if (pixel[1] - 1 >= 0) and (pixel[1] + 1 <= h - 1) and (pixel[0] - 1 >= 0) and (pixel[0] + 1 <= w - 1):
        return True
    else:
        return False


def extend_line(P1, P2 , offset = 100000):
    x1, y1 = P1
    x2, y2 = P2

    delta_x = x2 - x1
    delta_y = y2 - y1

    new_x1 = x1 - delta_x * offset
    new_y1 = y1 - delta_y * offset

    new_x2 = x2 + delta_x * offset
    new_y2 = y2 + delta_y * offset

    return ((new_x1, new_y1), (new_x2, new_y2))


def determine_side(P1, P2, P3):

    """Determines whether the point P3 is to left or to the right side of the line formed
       by the points P1 and P2
    """
    P1, P2 = extend_line(P1, P2)
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3

    d3 = (x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)

    # d1 is calculated for a point that we know lies on the left side of the line
    d1 = ((x1 - 1) - x1)*(y2 - y1) - (y1 - y1)*(x2 - x1)
    sign = lambda a: 1 if a > 0 else -1 if a < 0 else 0

    if sign(d3) == sign(d1):
        return "left"
    else:
        return "right"

def inner_angle(P1, P2, P3):
    """Computes the inner product formed by the lines generated from
       (P1(x1, y1), P2(x2, y2) and P2(x2, y2), P3(x3, y3))
       P2 is shared by both lines, hence it represents the point of ambiguity
    """
    side = determine_side(P1, P2, P3)
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3
    dx21 = x1 - x2
    dx31 = x3 - x2
    dy21 = y1 - y2
    dy31 = y3 - y2
    m12 = (dx21 * dx21 + dy21 * dy21) ** 0.5
    m13 = (dx31 * dx31 + dy31 * dy31) ** 0.5
    theta_radians = math.acos((dx21 * dx31 + dy21 * dy31) / (m12 * m13))
    theta_degrees = theta_radians * 180 / math.pi

    if side == "left":
        theta_degrees = 360 - theta_degrees

    return theta_degrees


def local_solver(P1, P2, neighbours):
    """
    from a set of neighbouring pixels selects the one with the minimum angular deviation
    from the direction given by the last two pixels of the stroke history (P1,P2).
    """
    minimum_angle = 100000
    selected_pixel = None

    for n in neighbours:
        delta_y, delta_x = delta[tuple(n)]
        P3 = (P2[0] + delta_x, P2[1] + delta_y)
        angle = inner_angle(P1, P2, P3)
        if angle < minimum_angle:
            selected_pixel = P3
            minimum_angle = angle
    return selected_pixel


def draw_strokes(image, strokes, colors=[(0,255,0)]):
    if len(image.shape)>2:
        h, w, d = image.shape
    else:
        h,w = image.shape
        d=3
    blank_image = np.zeros((h, w, d), np.uint8)
    color_index = 0
    for stroke in strokes:
        for p in stroke.points:
            # print('P:',p)
            cv2.circle(blank_image, p, 1, colors[color_index%len(colors)], -1)
        # cv2.imshow('1',blank_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
            #cv2.imwrite(f'{color_index}.bmp', blank_image)
        color_index += 1

    return blank_image


def distance(pixel_1, pixel_2):
    delta_x = (pixel_1[0] - pixel_2[0]) ** 2
    delta_y = (pixel_1[1] - pixel_2[1]) ** 2
    return (delta_x + delta_y)**0.5


def stroke_distance(stroke_1, stroke_2):
    """Returns the minimum distance between two strokes"""
    s_1 = stroke_1.starting_point
    e_1 = stroke_1.ending_point
    s_2 = stroke_2.starting_point
    e_2 = stroke_2.ending_point
    set_1 = [s_1, e_1]
    set_2 = [s_2, e_2]
    frontier = None
    minimum_distance = 1000000
    for p in set_1:
        for p2 in set_2:
            p_dist = distance(p, p2)
            if p_dist < minimum_distance:
                minimum_distance = p_dist
                frontier = (p,p2)
    return minimum_distance, frontier


def generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length = 10):
    """Generates an stroke until ambiguity, unless the current length is less than a predefined length threshold"""
    former_skeleton = skeleton_image.copy()
    ambiguity_pixel = None
    stroke_history = []
    stroke_history.append(pixel)
    all_possibilities = []
    skeleton_image[pixel[1], pixel[0]] = 0
    ambiguity_solved = True
    oth_p = []
    n1 = 0
    n2 = 0
    n3 = 0

    while (len(scan_8_pixel_neighbourhood(skeleton_image, pixel)) > 0) or not ambiguity_solved:
        if pixel[0]==211 and pixel[1]==207:
            print(11111111111)
            # testli(stroke_history,image,(0,255,0))

        #print(ambiguity_solved)
        if ambiguity_pixel:
            neighbours_ap = scan_8_pixel_neighbourhood(skeleton_image, ambiguity_pixel)
            #print("ambiguity pixel", ambiguity_pixel, "has", len(neighbours_ap), "neighbours")
            #print(neighbours_ap)
            if len(neighbours_ap) == 0:
                #print("hiiii", ambiguity_pixel)
                ambiguity_solved = True

        if len(scan_8_pixel_neighbourhood(skeleton_image, pixel)) == 0 and ambiguity_solved==False:
            # added

            skeleton_image[pixel[1], pixel[0]] = 0
            pixel = ambiguity_pixel
            all_possibilities.append(stroke_history)
            # testli(stroke_history, image, (0, 255, 0))
            stroke_history = []
            print('----------------------------------------------')

        # comparing with the new pixel
        neighbours = scan_8_pixel_neighbourhood(skeleton_image, pixel)
        print(len(neighbours))

            # added check
        #if len(neighbours) == 0:


        if len(neighbours) == 1:
            n1 +=1
            delta_y, delta_x = delta[tuple(neighbours[0])]
            pixel = (pixel[0] + delta_x, pixel[1] + delta_y)
            stroke_history.append(pixel)
            print("Stroke History == 1", stroke_history)
            skeleton_image[pixel[1], pixel[0]] = 0
        elif len(stroke_history) < minimum_length and len(neighbours) > 0:

            if len(stroke_history) < 2:
                print("neighbours XD XD", neighbours)
                print("neighbours XD", tuple(neighbours[0]))
                delta_y, delta_x = delta[tuple(neighbours[0])]
                pixel = (pixel[0] + delta_x, pixel[1] + delta_y)
                stroke_history.append(pixel)
                print("Stroke History > 0  and  <2", stroke_history)
                skeleton_image[pixel[1], pixel[0]] = 0
                oth_p.append(pixel)
                print('oth_p',oth_p)
            else:
                P1 = stroke_history[-2]
                P2 = stroke_history[-1]

                pixel = local_solver(P1, P2, tuple(neighbours))
                stroke_history.append(pixel)
                print("Stroke History  > 0  and  >>>>>>2", stroke_history)
                skeleton_image[pixel[1], pixel[0]] = 0
        else:
            n3 +=1
            #it is large enough so it can be compared later, hence we add it

            all_possibilities.append(stroke_history)
            # testli(stroke_history, image, (255, 111, 0))
            print("all_possibilities  --------------------不知道", stroke_history)
            stroke_history = []
            # we must go back to the original ambiguity
            if ambiguity_pixel:
                pixel = ambiguity_pixel
            else:
                ambiguity_pixel = pixel
                ambiguity_solved = False
                #print("ambiguity pixel", ambiguity_pixel, ambiguity_solved)
    # testli(stroke_history, image, (255, 255, 0))
    if len(stroke_history)>=9:
        all_possibilities.append(stroke_history)

    all_strokes = [Stroke(points, former_skeleton) for points in all_possibilities if len(points) > 4]
    # ima = np.zeros((image.shape), np.uint8)
    # for i in all_strokes:
    #
    #     testli(i.points,ima,(255,255,0))
    # for i in all_strokes:
    #     testli(i.points, image, (255, 0, 0))
    print('n1:{} n2:{} n3:{}'.format(n1,n2,n3))
    return all_strokes, former_skeleton , oth_p


# def generate_strokes2(skeleton_image):
#     all_strokes = []
#     while True:
#         pixel = find_top_left_most_pixel(skeleton_image, processing_index=0)
#         if  pixel == None:
#             break
#         try:
#             strokes, former_skeleton = generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length=10)
#             # if len(strokes) == 0:
#             #     break
#
#             for s in strokes:
#                 all_strokes.append(s)
#         except:
#             continue
#     return  all_strokes, former_skeleton


def generate_strokes2(skeleton_image):
    all_strokes = []
    wh_ci = 0
    while True:

        # 获得第一个坐标
        pixel = find_top_left_most_pixel(skeleton_image, processing_index=0)
        if  pixel == None:
            break

        strokes, former_skeleton,oth_p = generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length=10)
        # if len(strokes) == 0:
        #     break

        for s in strokes:
            all_strokes.append(s)
            wh_ci += 1
    print('wh_ci',wh_ci)
    return  all_strokes, former_skeleton,oth_p



def points_principal_component(points):
    """The points list must have  a length of 12"""
    x = np.array([p[0] for p in points])
    x_mean = x.mean()
    y = np.array([p[1] for p in points])
    y_mean  = y.mean()
    principal_component = np.sum((x - x_mean) * (y - y_mean))/(11)
    return principal_component


def best_stroke(former_stroke, possible_strokes ):
    """Returns best stroke to be merged inside possible_strokes according to the principal component"""
    best_stroke = None
    index = None
    minimum_difference = 10000000000000000
    pc_fs = points_principal_component(former_stroke.points[-12:])
    for index, ps in enumerate(possible_strokes):
        pc = points_principal_component(ps.points[0:12])
        diff = (pc_fs - pc)**2
        if diff < minimum_difference:
            minimum_difference = diff
            best_stroke = ps
    return best_stroke, index


def fill_stroke_gap(frontier):
    p1, p2 = frontier
    new_points_x = list(range(p1[0], p2[0] + 1, 1))
    new_points_y = list(range(p1[1], p2[1] + 1, 1))
    new_points = list(zip(new_points_x, new_points_y))
    return new_points


def alternative_single_merge(former_stroke, possible_strokes, image):
    """merges the former stroke with the best stroke within the possibilities"""
    possibilities = []
    strokes_to_be_erased = []


    for ps in possible_strokes:
        d,frontier = stroke_distance(former_stroke, ps)
        if d < 10:
            possibilities.append(ps)
    best_stroke_, index = best_stroke(former_stroke, possibilities)
    if best_stroke_:
        d, frontier = stroke_distance(former_stroke, best_stroke_)
        points_to_add = fill_stroke_gap(frontier)
        strokes_to_be_erased.append(best_stroke_.index)
        strokes_to_be_erased.append(former_stroke.index)
        #print(points_to_add)
        new_stroke = Stroke(best_stroke_.points + points_to_add + former_stroke.points, image)

        return new_stroke,strokes_to_be_erased
    else:
        return former_stroke,strokes_to_be_erased


def multiple_merge(all_strokes, image):

    former_stroke = all_strokes[0]
    to_compare = all_strokes[1:]
    # testli([to_compare],image,(255,255,0))
    while True:
        former_stroke, strokes_to_be_erased = alternative_single_merge(former_stroke, to_compare, image)
        to_compare = [tc for tc in to_compare if tc.index not in strokes_to_be_erased]
        if len(strokes_to_be_erased) == 0:
            break
    return former_stroke, to_compare


def generate_final_strokes(image):
    skeleton_image = getSkeleton(image)
    all_strokes, _ ,oth = generate_strokes2(skeleton_image)
    final_strokes = []
    while True:
        print("Generating Strokex")
        former_stroke, comparision_strokes = multiple_merge(all_strokes,image)
        if former_stroke:
            final_strokes.append(former_stroke)
        if len(comparision_strokes) == 0:
            break

        all_strokes = comparision_strokes
    return final_strokes

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

def get_maxmin(point_list):
    max_x = max(np.array(point_list)[:,:,0])
    max_y = max(np.array(point_list)[:,:,1])
    min_x = min(np.array(point_list)[:,:,0])
    min_y = min(np.array(point_list)[:,:,1])
    return max_x,max_y,min_x,min_y


def getDX_fw(contour_FW1,contour_FW2):
    x1 = contour_FW1[0] > contour_FW2[0]
    y1 = contour_FW1[1] > contour_FW2[1]
    x2 = contour_FW1[2] < contour_FW2[2]
    y2 = contour_FW1[3] < contour_FW2[3]
    return x1,y1,x2,y2
def getFw_list(all_contours):
    Fw  =[]
    for i in all_contours:
        max_x_1, max_y_1, min_x_1, min_y_1 = get_maxmin(i)
        Fw.append([max_x_1, max_y_1, min_x_1, min_y_1])
    return Fw


def  geiFw_Info(all_contours,Fw_list,save_path,image):
     # 所有的contours个数
     E = len(all_contours)
     nameNum = 0
     file_names= []
     del_file = []
     for i in range(E):
         new_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)
         nameNum+=1
         fwli  = get_maxmin(all_contours[i])
         bihua_FW_li = []
         bihua_FW_li.append(all_contours[i])
         houxu_fw = []

         for j in range(len(Fw_list)):

             max_x, max_y, min_x, min_y  = getDX_fw (fwli,Fw_list[j])

             if  max_x==True and max_y ==True and min_x ==True and min_y ==True:
                    houxu_fw.append(all_contours[j])
                    del_file.append(j+1)

         filename = save_path + str(nameNum) + '.jpg'


         cv2.drawContours(new_img, bihua_FW_li, -1, (255, 255, 255), -1)
         if len(houxu_fw) != 0:
            cv2.drawContours(new_img, houxu_fw, -1, (0, 0, 0), -1)
         cv2.imwrite( filename,new_img)
         file_names.append(filename)


     if len(houxu_fw)==(E-2):
            for i in file_names:
                file_names.remove(i)
                os.remove(i)
            cv2.drawContours(new_img, bihua_FW_li, -1, (255, 255, 255), -1)
            cv2.drawContours(new_img, houxu_fw, -1, (0, 0, 0), -1)
            cv2.imwrite(filename, new_img)
            file_names.append(filename)
     return file_names


def  getBlo_neighbour(qizhong_list,tmp_list,Skl_iamge):
    a= np.argwhere(np.array(qizhong_list)[:,1] ==max(np.array(qizhong_list)[:,1]))

    # testli11111([p1],img1,(255,255,0))
    if len(a)==1:
        # p = qizhong_list[int(a)]
        p = qizhong_list[int(a)]
        p1 = [p[0], p[1] + 1]
        n = neighbours(p1[0], p1[1], Skl_iamge)
        print(p1)
        print(n)
        print(tmp_list)
        p1 = neighbours_x_y(p[0], p[1])
        # print(p)
        # print(p1)
        for j in tmp_list:
            if p1[0]==j[0] and p1[1]==j[1]:
                print(' 邻域是:',j)

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

def get_allWP(skeleton11111111111111111111):
    h, w = skeleton11111111111111111111.shape
    p_li = []
    for i in range(h):
        for j in range(w):
            if skeleton11111111111111111111[i, j] == 255:
                p_li.append([i, j])
    return p_li


def get_jj_p(bh,fork_list):
    ned_x = bh[0]
    ned_y = bh[1]
    kaishi = neighbours_x_y(ned_x, ned_y)
    for i in kaishi:
        for j in fork_list:
            if i[0] == j[1] and i[1] == j[0]:
                    return i


def get_all_lianjie(list1,list2,jz_list,list_index,list2_index):
    qqqqqq = jz_list.copy()
    ceshi = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i]==None:
                    # qqqqqq[i][j] = 0
                    break
            if list2[j]==None:
                    # qqqqqq[i][j] = 0
                    break

            jl = distance(list1[i],list2[j])
            print(jl)
            if jl<=25:
                print('距离少于10，添加进入')
                print(i,j)
                qqqqqq[list_index][list2_index]=1
                # print('ceshi', list1[i], list2[j])
                ceshi.append(list1[i])
    print('ceshi',ceshi)
    # testli11111(ceshi,img,(255,0,0))
    return qqqqqq

def get_all_lianjie(list1,list2,jz_list,list_index,list2_index):
    qqqqqq = jz_list.copy()
    ceshi = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            if list1[i]==None:
                    # qqqqqq[i][j] = 0
                    break
            if list2[j]==None:
                    # qqqqqq[i][j] = 0
                    break

            jl = distance(list1[i],list2[j])
            print(jl)
            if jl<=15:
                print('距离少于10，添加进入')
                print(i,j)
                qqqqqq[list_index][list2_index]=1
                # print('ceshi', list1[i], list2[j])
                ceshi.append(list1[i])
    print('ceshi',ceshi)
    # testli11111(ceshi,img,(255,0,0))
    return qqqqqq

def get_jj_p2(bh,bihua_list):
    ned_x = bh[0]
    ned_y = bh[1]
    kaishi = neighbours_x_y(ned_x, ned_y)
    for i in kaishi:
        for j in range(len(bihua_list)):
            for k in range(len(bihua_list[j])):
                if i[0] == bihua_list[j][k][0] and i[1] == bihua_list[j][k][1]:
                    return i
def find_oth_jiaodian(d_a,d_b,d_c,d_d,bihua,al_bh):
    jl_list = []
    # 查找end点的 邻域  邻域和 fork点有交集的位置
    if len(d_a) < 3 and len(d_a) > 0:
        print('A')
        p = get_jj_p2(bihua[int(d_a[0])], al_bh)
        print('p',p)
        if p == None and len(d_a) > 1:
            p = get_jj_p2(bihua[int(d_a[1])], al_bh)
        jl_list.append(p)
    if len(d_b) < 3 and len(d_b) > 0:
        print('B')
        p = get_jj_p2(bihua[int(d_b[0])], al_bh)
        if p == None and len(d_b) > 1:
            p = get_jj_p2(bihua[int(d_b[1])], al_bh)
        jl_list.append(p)
        # jl_list['bh' + str(k)] = p
    if len(d_c) < 3 and len(d_c) > 0:
        print('C')
        p = get_jj_p2(bihua[int(d_c[0])], al_bh)
        print('11111111111111111111111', p)
        if p == None and len(d_c) > 1:
            p = get_jj_p2(bihua[int(d_c[1])], al_bh)
            print('2222222222222222222', p)
        if p == None:
            p = bihua[int(d_c[0])]
        jl_list.append(p)
        # jl_list['bh' + str(k)] = p
    if len(d_d) < 3 and len(d_d) > 0:
        print('D')
        p = get_jj_p2(bihua[int(d_d[0])], al_bh)
        print('11111111111111111111111DDD', p)
        if p == None and len(d_d) > 1:
            p = get_jj_p2(bihua[int(d_d[1])], al_bh)
        jl_list.append(p)
    return jl_list

# fork_list:      所有的交叉点
# all_bihua_list :笔画所有点位

def getFork_posion(fork_list,all_bihua_list):

    jl_list = []
    ceshi_lisgt  = {}
    k = 0
    for bihua in all_bihua_list:
        # 所有笔画的x  和y 的集合
        y1_x = np.array(bihua)[:, 0]
        y1_y = np.array(bihua)[:, 1]
        # 最大的x  y 最小的x y
        y1_x_max = max(y1_x)
        y1_y_max = max(y1_y)
        y1_x_min = min(y1_x)
        y1_y_min = min(y1_y)
        # 找出  最高点   最低点  最左点 最右点 的个数  超过1的说明有不是该定点  不具备唯一性
        a = np.argwhere(y1_x == y1_x_max)
        b = np.argwhere(y1_x == y1_x_min)
        c = np.argwhere(y1_y == y1_y_max)
        d = np.argwhere(y1_y == y1_y_min)
        print('a',len(a))
        print('b',len(b))
        print('c',len(c))
        print('d',len(d))

        # 查找end点的 邻域  邻域和 fork点有交集的位置
        if len(a)<2 and len(a)>0:
            print('A')
            p = get_jj_p(bihua[int(a[0])], fork_list)
            if p==None and  len(a)>1:
                p = get_jj_p(bihua[int(a[1])], fork_list)
            jl_list.append(p)
        if len(b)<2 and len(b)>0:
            print('B')
            p = get_jj_p(bihua[int(b[0])], fork_list)
            if p == None and  len(b)>1:
                p = get_jj_p(bihua[int(b[1])], fork_list)
            jl_list.append(p)
            # jl_list['bh' + str(k)] = p
        if len(c)<2 and len(c)>0:
            print('C')
            p = get_jj_p(bihua[int(c[0])], fork_list)
            print('11111111111111111111111',p)
            if p == None and len(c)>1:
                p = get_jj_p(bihua[int(c[1])], fork_list)
                print('2222222222222222222', p)
            if p==None:
                p = bihua[int(c[0])]
            jl_list.append(p)
            # jl_list['bh' + str(k)] = p
        if len(d)<2 and len(d)>0:
            print('D')
            p = get_jj_p(bihua[int(d[0])], fork_list)
            print('11111111111111111111111DDD', p)
            if p == None and  len(d)>1:
                p = get_jj_p(bihua[int(d[1])], fork_list)
            jl_list.append(p)
            # jl_list['bh' + str(k)] = p
        if jl_list==[None]:
            print('123123123123123123123123123')
            jl_list = find_oth_jiaodian(a,b,c,d,bihua,all_bihua_list)
        ceshi_lisgt['bh' + str(k)] = jl_list
        jl_list= []
        k += 1


    # ceshi_list 是笔画 和 fork有交集的位置 可以理解为 他和其他位置有交集

    print('ceshi_lisgt:   ',ceshi_lisgt)
    keys = list(ceshi_lisgt.keys())
    q111111 = np.zeros([len(keys), len(keys)])
    for i in range(len(keys)):
        for j in range(len(keys)):
             if i !=j:
                 # print(i,j)
                 list1 = ceshi_lisgt.get(keys[i])
                 list2 = ceshi_lisgt.get(keys[j])
                 print(list1)
                 # testli11111(list1,img,(255,0,0))
                 # testli11111(list2,img,(255,0,0))
                 if i == 0 and j ==1:
                     print(1)
                 q111111 = get_all_lianjie(list1,list2,q111111,i,j)
    return q111111,ceshi_lisgt

def get_strokes_Weight(all_strokes, ):
    weight_list = []

    for i in all_strokes:
        A_x = 0
        A_y = 0
        point_list = np.array(i.points)
        x_list = point_list[:, 0]
        y_list = point_list[:, 1]
        point_max_x = max(x_list)
        point_max_y = max(y_list)
        point_min_x = min(x_list)
        point_min_y = min(y_list)
        kd = (point_max_x + 1) - point_min_x
        gd = (point_max_y + 1) - point_min_y
        for i in range(point_min_x, point_max_x + 1):
            A_x += i

        for j in range(point_min_y, point_max_y + 1):
            A_y += j
        A_x = A_x / (kd)
        A_y = A_y / (gd)
        weight_list.append([A_x, A_y])

    return weight_list

all_p = []
all_p1 = []

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
img = cv2.imread(files_name[4])
img = 255-img
showImage(img)
skeleton_image ,ceshi = getSkeleton(img)
p_list = get_allWP(skeleton_image)
skeleton_image_cy= skeleton_image.copy()
showImage(skeleton_image_cy)
tmp = []
for i in ceshi:
    a = neighbours(i[0],i[1],skeleton_image)
    k = np.argwhere(a)
    if len(k)>=3:
        for j in k:
            tmp.append(neighbours_x_y(i[1],i[0])[int(j)])
            # skeleton_image[neighbours_x_y(i[1], i[0])[int(j)]]==0
sk = testli(tmp, skeleton_image, (0, 0, 0))
print('',tmp)
# showImage(sk)
all_strokes, _, oth_p = generate_strokes2(skeleton_image)
bihua_all_list = []
for i in all_strokes:
    bihua_all_list.append(i.points)
bihua_lines = []
for i in all_strokes:
    bihua_lines.append([i.points[0],i.points[-1]])

    # testli11111([i.points[0],i.points[-1]],img1,(255,0,0))
import util
fork = util.getfork(tmp)
intersection = util.get_stroke_intersection(fork,bihua_lines)
# print(intersection)
lianxuxing = {}
q = np.zeros((len(fork),len(bihua_lines)))
# threshold = cv2.bitwise_not(img)
# threshold[threshold<20]=255
# threshold[threshold>22]=0
img = 255-img

# showImage(img)
# testli11111([fork.get('3')[0]],img,(255,255,0))
gary_img = cv2.imread(files_name[4],0)

bihua_ins_lines = []
for i in all_strokes:
    bihua_ins_lines.append([i.points[int(len(i.points)/3)],i.points[int(len(i.points)/2)]])
keys = fork.keys()
for j in keys:
    for i in intersection:
        if i[1]==j:
            q[int(j),i[0]]=1
qqqqqq = []
for i in range(len(q)):
    jd_list = []
    for j in range(len(q[0])):
        if q[i,j]==1:
            Dis , bo_p_list ,jd  = util.get_PBOD(fork.get(str(i))[0],gary_img,bihua_ins_lines[j])
            util.show_image(Dis)
            jd_list.append([jd,j])
            # indexes = util.get_peak(Dis)
    qqqqqq.append(jd_list)
print('11111111111',qqqqqq)
print('11111111111')
for d  in range(len(qqqqqq)):
    for i in range(len(qqqqqq[d])):
        for j in range(len(qqqqqq[d])):
            if qqqqqq[d][i][0]-qqqqqq[d][j][0]>170  and qqqqqq[d][i][0]-qqqqqq[d][j][0] <190:
                 print('第{}组的  {}   {}  位置'.format(d,qqqqqq[d][i][1],qqqqqq[d][j][1]))
if len(qqqqqq)==1:
    print(11111)
    print('只有一个fork点 所以他们为一笔')



# print(bihua_lines[3])
# print(bihua_lines[5])
# print(bihua_lines[6])
# a = fork.get('2')
# print(len(a))
# testli11111(a[1],img,(255,0,0))
# testli11111(bihua_lines[6],img,(255,0,0))
# print(indexes)
# print(Dis)
# print(bo_p_list)
# util.show_image(Dis)

# z1 = bo_p_list[indexes[0]]
# z2 = bo_p_list[indexes[1]]
# z3 = bo_p_list[indexes[2]]
# print(z1)
# print(z2)
# print(z3)
# print(bihua_lines[3])
# print(bihua_lines[5])
# print(bihua_lines[6])
# testli11111([z1],img,(255,0,0))
# testli11111([z2],img,(255,0,0))
# testli11111([z3],img,(255,0,0))
# keys = fork.keys()
# for j in keys:
#     for i in intersection:
#         if i[1]==j:
#             q[int(j),i[0]]=1
# test_iamge = cv2.imread('images/test2.png')
# testli11111(fork.get('1'),test_iamge,(255,0,0))
# testli11111(bihua_all_list[1],test_iamge,(255,255,0))
# testli11111(bihua_all_list[5],test_iamge,(0,255,0))
# testli11111(bihua_all_list[2],test_iamge,(0,0,255))








#     img = cv2.imread('images/test2.png')
#     print(i)
#     testli11111(bihua_lines[i[0]], img, (255, 0, 0))
#     testli11111(fork.get(i[1]), img, (255, 0, 255))
