import random
import math

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import util
import libs.tony_beltramelli_detect_peaks as tony_beltramelli_detect_peaks
import libs.findpeaks
import Strokes_Type as st
import copy
#key: row and column of the neighbouring pixels, value: the amount to be added to the central pixel -> h,wpl.,
delta = {(0,0): (-1,-1), (0,1): (-1,0), (0,2): (-1,1),
            (1,0): (0,-1), (1,1): (0,0), (1,2): (0,1),
            (2, 0): (1, -1), (2,1): (1,0), (2,2): (1,1)}

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (80, 127, 255), (255, 0, 255),
          (255, 255, 0), (96, 164, 244)]



def testli11111(oth,image,ys):
    print(oth)
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

        real_ending_points = self.far_away_end_points(ending_points)
        return (real_ending_points[0], real_ending_points[-1])


def threshold_image(image, min_p = 200, max_p = 255):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayscaled, min_p, max_p, cv2.THRESH_BINARY)
    return threshold


def getSkeleton (image):
    # Applies a skeletonization algorithm, thus, transforming all the strokes inside the sketch into one pixel width lines
    threshold = threshold_image(image)
    threshold = cv2.bitwise_not(threshold)
    threshold[threshold == 255] = 1
    skeleton = skeletonize(threshold)
    skeleton = img_as_ubyte(skeleton)
    return skeleton

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

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    # if  x=10.y=10   3点钟方向 逆时针 点位
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    # testli11111([[x,y1] ,[x_1,y1], [x_1,y],  [x_1,y_1], [x,y_1] ,[x1,y_1], [x1,y], [x1,y1]],img,(255,0,0))
                 # p0        p1            p2               p3          p4              p5          p6             p7
    return [img[y1][x] ,img[y1][x_1], img[y][x_1],  img[y_1][x_1], img[y_1][x] , img[y_1][x1],  img[y][x1], img[y1][x1]]


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
        # testli11111(stroke.points,image,(255,0,0))
        for p in stroke.points:
            cv2.circle(blank_image, p, 1, colors[color_index%len(colors)], -1)
            #cv2.imwrite(f'{color_index}.bmp', blank_image)
        color_index += 1

    return blank_image


def distance(pixel_1, pixel_2):
    delta_x = (pixel_1[0] - pixel_2[0])**2
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

def stroke_distance2(stroke_1, stroke_2):
    """Returns the minimum distance between two strokes"""
    s_1 = stroke_1.starting_point
    e_1 = stroke_1.ending_point
    s_2 = stroke_2.starting_point
    e_2 = stroke_2.ending_point
    set_1 = [s_1, e_1]
    set_2 = [s_2, e_2]
    frontier = None
    max_distance = 1
    for p in set_1:
        for p2 in set_2:
            p_dist = distance(p, p2)
            if p_dist > max_distance:
                max_distance = p_dist
                frontier = (p,p2)
    return max_distance, frontier


def generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length = 10):
    """Generates an stroke until ambiguity, unless the current length is less than a predefined length threshold"""
    former_skeleton = skeleton_image.copy()
    ambiguity_pixel = None
    stroke_history = []
    stroke_history.append(pixel)
    all_possibilities = []
    skeleton_image[pixel[1], pixel[0]] = 0
    ambiguity_solved = True

    while (len(scan_8_pixel_neighbourhood(skeleton_image, pixel)) > 0) or not ambiguity_solved:
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
            stroke_history = []

        # comparing with the new pixel
        neighbours = scan_8_pixel_neighbourhood(skeleton_image, pixel)


            # added check
        #if len(neighbours) == 0:


        if len(neighbours) == 1:
            delta_y, delta_x = delta[tuple(neighbours[0])]
            pixel = (pixel[0] + delta_x, pixel[1] + delta_y)
            stroke_history.append(pixel)
            print("Stroke History", stroke_history)
            skeleton_image[pixel[1], pixel[0]] = 0
        elif len(stroke_history) < minimum_length and len(neighbours) > 0:
            if len(stroke_history) < 2:
                print("neighbours XD XD", neighbours)
                print("neighbours XD", tuple(neighbours[0]))
                delta_y, delta_x = delta[tuple(neighbours[0])]
                pixel = (pixel[0] + delta_x, pixel[1] + delta_y)
                stroke_history.append(pixel)
                print("Stroke History", stroke_history)
                skeleton_image[pixel[1], pixel[0]] = 0
            else:
                P1 = stroke_history[-2]
                P2 = stroke_history[-1]

                pixel = local_solver(P1, P2, tuple(neighbours))
                stroke_history.append(pixel)
                print("Stroke History", stroke_history)
                skeleton_image[pixel[1], pixel[0]] = 0
        else:
            #it is large enough so it can be compared later, hence we add it

            all_possibilities.append(stroke_history)
            print("all_possibilities", stroke_history)
            stroke_history = []
            # we must go back to the original ambiguity
            if ambiguity_pixel:
                pixel = ambiguity_pixel
            else:
                ambiguity_pixel = pixel
                ambiguity_solved = False
                #print("ambiguity pixel", ambiguity_pixel, ambiguity_solved)

    if len(stroke_history)>=1:
        all_possibilities.append(stroke_history)

    all_strokes = [Stroke(points, former_skeleton) for points in all_possibilities if len(points) > 3]

    return all_strokes, former_skeleton


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
    while True:
        pixel = find_top_left_most_pixel(skeleton_image, processing_index=0)
        if  pixel == None:
            break

        strokes, former_skeleton = generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length=1)
        # if len(strokes) == 0:
        #     break

        for s in strokes:
            all_strokes.append(s)

    return  all_strokes, former_skeleton



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
            # testli11111(former_stroke.points, image, (255, 0, 0))
            # testli11111(ps.points, image, (255, 0, 0))
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

    testimage = image.copy()
    testimage = testimage[:, :, 0]
    testimage = 255-testimage

    for ps in possible_strokes:
        d,frontier = stroke_distance(former_stroke, ps)

        if d < 10:
            d2 ,frontier2 = stroke_distance2(former_stroke,ps)
            pbod1 = util.get_PBOD2(frontier[0],testimage,frontier2[0])
            pbod2 = util.get_PBOD2(frontier[0],testimage,frontier2[1])
            if len(pbod1)>2 and  len(pbod2)>2:
                if abs(pbod1[2]-pbod2[2])<200 and abs(pbod1[2]-pbod2[2])>150:
                    possibilities.append(ps)
            # util.show_image(pbod[0])
            # possibilities.append(ps)

    # print('possibilities11111111111111111111111111111111',len(possibilities))
    # testli11111(possibilities.points,image,(255,0,0))
    best_stroke_, index = best_stroke(former_stroke, possibilities)
    if best_stroke_:
        d, frontier = stroke_distance(former_stroke, best_stroke_)
        points_to_add = fill_stroke_gap(frontier)
        strokes_to_be_erased.append(best_stroke_.index)
        strokes_to_be_erased.append(former_stroke.index)
        #print(points_to_add)
        new_stroke = Stroke(best_stroke_.points + points_to_add + former_stroke.points, image)
        # testli11111(points_to_add,image,(255,0,0))
        return new_stroke,strokes_to_be_erased
    else:
        return former_stroke,strokes_to_be_erased


def multiple_merge(all_strokes, image):

    former_stroke = all_strokes[0]
    to_compare = all_strokes[1:]
    while True:
        former_stroke, strokes_to_be_erased = alternative_single_merge(former_stroke, to_compare, image)
        to_compare = [tc for tc in to_compare if tc.index not in strokes_to_be_erased]
        if len(strokes_to_be_erased) == 0:
            break
    return former_stroke, to_compare

def getall_jd(all_strokes):
    jd_list = []
    for index1 , stroke1 in enumerate(all_strokes):
        for index2 , stroke2 in enumerate(all_strokes):
            if index1!=index2:
                p1_1 = stroke1.starting_point
                p1_2 = stroke1.ending_point
                p2_1 = stroke2.starting_point
                p2_2  = stroke2.ending_point
                p_i = [p1_1,p1_2]
                p_j = [p2_1,p2_2]
                for i in p_i:
                    for j in p_j:
                        if i !=j:
                            p_dist = distance(i, j)
                            if p_dist<10:
                                jd_list.append([index1,index2,i])
    return  jd_list

def getall_jiaodu(list_1,list2):
    if list_1!=list2:
        p1_1 = list_1[0]
        p1_2 = list_1[1]
        p2_1 = list2[0]
        p2_2  = list2[1]
        p_i = [p1_1,p1_2]
        p_j = [p2_1,p2_2]
        for i in p_i:
            for j in p_j:
                if i-j>160 and i-j<200:
                    return True




def generate_final_strokes(image):
    skeleton_image = getSkeleton(image)
    all_strokes, _ = generate_strokes2(skeleton_image)
    # jd_list = getall_jd(all_strokes)
    # undigraph = get_undigraph(jd_list, all_strokes)
    # 判断所有笔画180波峰的  如果是则两个为一笔画  180度规则
    # new_allstorkes = stroket_180_relu(all_strokes, undigraph, jd_list)
    # jd_list = getall_jd(new_allstorkes)
    # jd_list = util.remove_list_repetition(jd_list)
    # new_allstorkes = stroket_start2end_relu(new_allstorkes, jd_list)
    # jd_list = getall_jd(new_allstorkes)
    # jd_list = util.remove_list_repetition(jd_list)
    # new_allstorkes = stroket_end2there(new_allstorkes, jd_list)

    final_strokes = []
    while True:
        print("Generating Strokex")
        former_stroke, comparision_strokes = multiple_merge(all_strokes,image)
        # former_stroke, comparision_strokes = multiple_merge(new_allstorkes,image)
        if former_stroke:
            final_strokes.append(former_stroke)
        if len(comparision_strokes) == 0:
            break

        all_strokes = comparision_strokes
        # new_allstorkes = comparision_strokes

    return final_strokes

def  get_undigraph(list_jd ,all_stroket):
    qqqq = np.zeros((len(all_stroket), len(all_stroket)))
    for i in list_jd:
        qqqq[i[0], i[1]] = 1
    return qqqq

def findlist_same(di_t,mu_po_list):
    if  len(mu_po_list)==0:
       return True
    elif len(di_t)==0:
        return False
    else:
        di = di_t.copy()
        keys = di.keys()
        for i in keys:
            if (di.get(i)[0][0] == mu_po_list[0][0] and  di.get(i)[0][1] == mu_po_list[0][1]) or \
                    (di.get(i)[0][1] == mu_po_list[0][0] and  di.get(i)[0][0] == mu_po_list[0][1]) or \
                    (di.get(i)[0][0] == mu_po_list[0][1] and  di.get(i)[0][1] == mu_po_list[0][0]):
                return True
            else:
                return False
def chongfuquchu(chongfu_list,all_Strokes,te_image):

    chongfu_list= list(chongfu_list)
    mu_po = {}
    for index1, i in enumerate(chongfu_list):
        mu_po_list = []
        for index2, j in enumerate(chongfu_list):
            if index1 != index2:
                if i[0] == j[1] or i[0] == j[0]:
                    te_image =  te_image.copy()
                    mu_po_list.append([index1, index2])
        if mu_po_list != [] and len(mu_po_list) > 0 and findlist_same(mu_po, mu_po_list) == False:
            mu_po[str(index1)] = mu_po_list
    if len(mu_po)!=0:
        for key in mu_po.keys():
            zhenghe = []
            record_remove_list = []
            for i in mu_po.get(key):
                for j in i:
                    zhenghe.append(chongfu_list[j][0])
                    zhenghe.append(chongfu_list[j][1])
                    record_remove_list.append(j)
            record_remove_list = list(set(record_remove_list))
            record_remove_list.sort(reverse=True)
            for i in record_remove_list:
                chongfu_list.remove(chongfu_list[i])
            zhenghe = list(set(zhenghe))
            zhenghe.sort(reverse=True)
            poins = []
            for i in zhenghe:
                poins += all_Strokes[i].points
            new_stroke = Stroke(poins, image)
            all_Strokes.append(new_stroke)
            zhenghe.sort(reverse=True)
            for i in zhenghe:
                all_Strokes.pop(i)
            chongfu_list = np.array(chongfu_list)
            chongfu_list[chongfu_list >= min(zhenghe)] = chongfu_list[chongfu_list >= min(zhenghe)] - len(zhenghe)
            chongfu_list = chongfu_list.tolist()
            mu_po.pop(key)
            return all_Strokes,chongfu_list,mu_po
    else:
        return all_Strokes,None,None



def stroket_180_relu(all_strokets,undigraph,jd_list,image):
    # image = threshold_image(image)
    jlzb = []
    for index, undigraph_i in enumerate(undigraph):
        n = np.argwhere(undigraph_i == 1)
        if len(n) != 0:
            # te_image = cv2.imread('testline/58114.png')
            for n_i in n:
                for j in jd_list:
                    if j[0] == index and j[1] == int(n_i):
                        # sz = util.get_strokes_Weight(all_strokets,[index,int(n_i)])
                        sz = stroke_distance2(all_strokets[index],all_strokets[int(n_i)])
                        # teimage = image.copy()
                        jd = cal_ang(sz[1][0],j[2],sz[1][1])
                        if len(np.argwhere(np.array(jd)>=160))>0 \
                                and len(np.argwhere(np.array(jd)<=200))>0 :

                            jlzb.append([index, int(n_i)])



    if len(jlzb)!=0:
        jlzb = util.remove_list_repetition(jlzb)
        all_strokets,jlzb,keys = chongfuquchu(jlzb,all_strokets,image)
        while  jlzb!=None or keys!=None:
            all_strokets,jlzb, keys = chongfuquchu(jlzb, all_strokets, image)
        while type(all_strokets[0])==list:
            all_strokets = all_strokets[0]
        return all_strokets,jlzb
    else:
        while type(all_strokets[0])==list:
            all_strokets = all_strokets[0]
        return all_strokets,jlzb,None



def stroket_180_relu2(all_strokets,undigraph,jd_list,image):
    image = threshold_image(image)
    jlzb = []
    for index, undigraph_i in enumerate(undigraph):
        n = np.argwhere(undigraph_i == 1)
        if len(n) != 0:
            te_image = cv2.imread('testline/58117.png')
            for n_i in n:
                for j in jd_list:
                    if j[0] == index and j[1] == int(n_i):
                        sz = util.get_strokes_Weight(all_strokets,[index,int(n_i)])
                        jd = cal_ang(sz[0],j[2],sz[1])
                        if len(np.argwhere(np.array(jd)>=170))>0:
                            jlzb.append([index, int(n_i)])

    if len(jlzb)!=0:
        jlzb = util.remove_list_repetition(jlzb)
        for index1,i in enumerate(jlzb):
            for index2,j in enumerate(jd_list):
                # if index1!=index2:
                    if i[0] == j[0] and i[1] == j[1]:
                        new_stroke = Stroke(all_strokets[i[0]].points + [j[2]] + all_strokets[i[1]].points, image)
                        all_strokets.append(new_stroke)
        jihe = []
        for i in jlzb:
            jihe.append(i[0])
            jihe.append(i[1])
        jihe = list(set(jihe))
        jihe.sort(reverse=True)
        for i in jihe:
            all_strokets.pop(i)
        while type(all_strokets[0])==list:
            all_strokets = all_strokets[0]

        return all_strokets,jlzb
    else:
        while type(all_strokets[0])==list:
            all_strokets = all_strokets[0]
        return all_strokets,jlzb,None


# 角度计算
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
    try:
        a_1 = (a * a - b * b - c * c) / (-2 * b * c)
        if a_1 > 1:
            a_1 = 1
        elif a_1 < -1:
            a_1 = -1
        A = math.degrees(math.acos(a_1))
    except ZeroDivisionError:
        A = 0
    try:
        b_1 = (b * b - a * a - c * c) / (-2 * a * c)
        if b_1>1:
            b_1 = 1
        elif b_1<-1:
            b_1=-1
        B = math.degrees(math.acos(b_1))
    except ZeroDivisionError:
        B = 0
    try:
        c_1 = (c * c - a * a - b * b) / (-2 * a * b)
        if c_1 > 1:
            c_1 = 1
        elif c_1 < -1:
            c_1 = -1
        C = math.degrees(math.acos(c_1))
    except ZeroDivisionError:
        C = 0

    return A,B,C



# 判断一
# 两两相连 且 链接处 另外一端为为独立笔画  且无第三笔画  那么两者为 相连
def stroket_start2end_relu(new_allstorkes,jd_list):
    jlzb = []
    for i in jd_list:

        q = 0
        w = 0
        for j in jd_list:
            if i != j:
                if i[0] == j[0] or i[0] == j[1]:
                    q += 1
                if i[1] == j[0] or i[1] == j[1]:
                    w += 1
        if q < 1 and w < 1:
            new_stroke = Stroke(new_allstorkes[i[0]].points + [i[2]] + new_allstorkes[i[1]].points, image)
            new_allstorkes.append(new_stroke)
            jlzb.append([i[0],i[1]])
    jihe = []
    for i in jlzb:
        jihe.append(i[0])
        jihe.append(i[1])
    jihe.sort(reverse=True)
    for i_j in jihe:
        new_allstorkes.pop(i_j)
    return new_allstorkes



def  stroket_end2there(new_allstorkes,jd_list,image):
    ds, ds2 = util.get_jiaodian(jd_list)
    dd = []
    ds21 = []
    if len(ds2) != 0:
        ds21.extend(ds2)
    while True:
        if len(dd) == len(ds) or len(dd) == 1:
            break
        else:
            ds, ds2 = util.get_jiaodian(ds)
            dd = ds
            print('len', len(ds2))
            if len(ds2) != 0:
                ds21.extend(ds2)
    dd = list(set(dd[0]))
    p = None
    for i in dd:
        if type(i) == tuple:
            p = i
            dd.remove(i)
    fz = []
    for i in dd:
        jd1 = util.get_PBOD4(p, image, new_allstorkes[i].points[int(len(new_allstorkes[i].points) / 8)])
        jd2 = util.get_PBOD4(p, image, new_allstorkes[i].points[int(len(new_allstorkes[i].points) / 8) * 7])
        fz.append([jd1, jd2])

    jlzb = []
    for i in range(len(fz)):
        for j in range(len(fz)):
            if i != j:
                s = getall_jiaodu(fz[i], fz[j])
                if s == True:
                    new_stroke = Stroke(new_allstorkes[dd[i]].points + [p] + new_allstorkes[dd[j]].points, image)
                    new_allstorkes.append(new_stroke)
                    jlzb.append([dd[i], dd[j]])
    jihe = []
    for i in jlzb:
        jihe.append(i[0])
        jihe.append(i[1])
    jihe.sort(reverse=True)
    for i_j in jihe:
        new_allstorkes.pop(i_j)
    return new_allstorkes


def remove_liite_stroke(all_strokes):
    jilu = []
    for i in all_strokes:
        if len(i.points)<5:
            jilu.append(i.index)

    jilu.sort(reverse=True)
    for i_j in jilu:
        all_strokes.pop(i_j)
    return all_strokes

image = cv2.imread('testline/31968.png')
# showImage(image)
skeleton_image = getSkeleton(image)
# cp_skeleton_image =copy.deepcopy(skeleton_image)
# showImage(skeleton_image)
all_strokes, _ = generate_strokes2(skeleton_image)


# print('all_strokes:',len(all_strokes))
# for i in all_strokes:
#     testli11111(i.points,image,colors[random.randint(0,7)])



js = 0
jd_list = getall_jd(all_strokes)
jd_list = util.remove_list_repetition(jd_list)
undigraph = get_undigraph(jd_list, all_strokes)
new_allstorkes = stroket_180_relu(all_strokes, undigraph, jd_list, image)
ji = new_allstorkes[1]
# for i in new_allstorkes[0]:
#     testli11111(i.points,image,colors[random.randint(0,7)])
while True:
    jd_list = getall_jd(new_allstorkes[0])
    jd_list = util.remove_list_repetition(jd_list)
    undigraph = get_undigraph(jd_list, new_allstorkes[0])
    new_allstorkes = stroket_180_relu(new_allstorkes[0], undigraph, jd_list, image)
    li = new_allstorkes[1]
    print(js)
    if len(new_allstorkes)>2 or li==ji :
        new_allstorkes =new_allstorkes
        break
    js+=1

# for i in new_allstorkes[0]:
#     testli11111(i.points,image,colors[random.randint(0,7)])
js = 0
jd_list = getall_jd( new_allstorkes[0])
jd_list = util.remove_list_repetition(jd_list)
undigraph = get_undigraph(jd_list, new_allstorkes[0])
new_allstorkes = stroket_180_relu2( new_allstorkes[0],undigraph,jd_list,image)
ji = new_allstorkes[1]
# for i in new_allstorkes[0]:
#     testli11111(i.points,image,colors[random.randint(0,7)])
while True:
    jd_list = getall_jd(new_allstorkes[0])
    jd_list = util.remove_list_repetition(jd_list)
    undigraph = get_undigraph(jd_list, new_allstorkes[0])
    new_allstorkes = stroket_180_relu2(new_allstorkes[0], undigraph, jd_list, image)
    li = new_allstorkes[1]
    print(js)
    if len(new_allstorkes)>2 or li==ji :
        new_allstorkes = new_allstorkes
        break
    js+=1

# for i in new_allstorkes[0]:
#     testli11111(i.points,image,colors[random.randint(0,7)])
jd_list = getall_jd(new_allstorkes[0])
jd_list = util.remove_list_repetition(jd_list)

# 重新做下这个   观点     几个比划相连   两端的尾垫  都没有和其他笔画相连












new_allstorkes = stroket_start2end_relu(new_allstorkes[0],jd_list)
arr = []
for i in new_allstorkes:
    point_jihe = []
    for j in i.points:
        if j != i.ending_point:
            point_jihe.append([j[0],j[1],0])
        else:
            point_jihe.append([j[0],j[1],1])
    point_jihe =np.array(point_jihe)
    arr.extend(point_jihe)
arr = np.array(arr)
np.save('arr.npy',arr)
    # testli11111(i.points,image,colors[random.randint(0,7)])



# import Diluting
# testli11111(new_allstorkes[3].points,image,colors[random.randint(0,7)])
# dl = Diluting.LimitVerticalDistance()
# dd = Diluting.DouglasPeuker()
# szu =dl.diluting(new_allstorkes[3].points)








# sz = dd.main(new_allstorkes[3].points)
# testli11111(szu,image,colors[random.randint(0,7)])
# testli11111(sz,image,colors[random.randint(0,7)])




# tony_beltramelli_detect_peaks.detect_peaks()
# a = findpeakssssssss(new_allstorkes[0].points)
# print(a)






# jd_list = getall_jd(new_allstorkes)
# jd_list = util.remove_list_repetition(jd_list)
# new_allstorkes = stroket_end2there(new_allstorkes,jd_list,image)
# for i in new_allstorkes:
#     testli11111(i.points,image,colors[random.randint(0,7)])
# jd_list = getall_jd(new_allstorkes)
# jd_list = util.remove_list_repetition(jd_list)
# print(jd_list)







# for i in new_allstorkes:
#     testli11111(i.points,image,colors[random.randint(0,7)])
# 尾部或者头部有链接但是没有后续的
# 循环遍历所有笔画


#     # 如果有链接则暂时放弃
#     # 如果无连接判断另外一端是否有单链接 如果有则他们一笔画
#     # 如果有多个链接则判断是否是在PBPD的180°波峰

# for i in jd_list:



# for i in new_allstorkes:
#     testli11111(i.points, image, colors[random.randint(0, 7)])
# before = draw_strokes(image, all_strokes, colors)
# showImage(before)









# all_strokes[0].starting_point
# all_strokes[0].ending_point
# print(len(all_strokes))


# before = draw_strokes(image, all_strokes, colors)
# displaying contours before merge
# showImage(before)
#
#
#
# final_strokes = generate_final_strokes(image)
# print(len(final_strokes))
# displaying contours after merge
# after = draw_strokes(image, final_strokes, colors)
# showImage(after)

