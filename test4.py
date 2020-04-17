import numpy as np
import cv2
import Strokes_Type

ls = [[]]
class Point(object):

    def __init__(self,x,y):
        self.x = x
        self.y = y
    def y(self):
        return self.y

    def x(self):
        return self.x

if __name__ == '__main__':
    img = cv2.imread('images/test2.png')
    imOrg = np.zeros((img.shape),img.dtype)
    imOrg.fill(255)
    cv2.circle(imOrg,(100,100),3,(255,0,0))
    cv2.circle(imOrg,(100,200),3,(255,0,0))
    cv2.circle(imOrg,(200,200),3,(255,0,0))
    cv2.circle(imOrg,(200,180),3,(255,0,0))
    # cv2.imshow('10',imOrg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    p1 = Point(100, 100)
    print(p1.x)
    d_list = [Point(100, 100), Point(200, 140)]
    fork_list = [Point(200, 120)]
    s = Strokes_Type.Strokes_type(d_list,fork_list)
    # b = s.rule_point()
    # a  = s.rule_vertical_hook()
    # b  = s.rule_across_vertical_lift()
    c = s.rule_across_hook()
    # print(a)
    # print(b)
    print(c)