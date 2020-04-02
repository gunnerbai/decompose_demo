import numpy as np
import cv2



class Strokes_type(object):


    def __init__(self,points,fork_list):
            self.points     =   points
            self.fork_list  =   fork_list


    def find_all_type(self):

        return True
    # 点  只有一个点没有交集的单独笔画
    def rule_point(self):
        return 1
    # 横  一条横向的直线   AB 两点   A B 在横向 约180的方向
    def rule_across(self):
        A = self.points[0]
        B = self.points[-1]
        if B.x > A.x and  B.y - A.y <30:
            return True
        else:
            return False
    # 竖  一个竖向的直线  AB 两点   A B 在竖向 约180的方向
    def rule_vertical(self):
        A = self.points[0]
        B = self.points[-1]
        if B.x - A.x < 30 and B.y > A.y:
            return True
        else:
            return False
    # 撇    AB 两点   B点在A点左下方的位置  且 为 A-》B的向量
    def rule_neglect(self):
        A = self.points[0]
        B = self.points[-1]
        if B.x < A.x and B.y > A.y :
            return True
        else:
            return False
    # 捺      AB 两点   B点在A点右下方的位置  且 为 A-》B的向量
    def rule_restrain(self):
        A = self.points[0]
        B = self.points[-1]
        if B.x > A.x and B.y > A.y:
            return True
        else:
            return False

    # 提    AB 两点   B点在A点左下方的位置     且 为 A《-B的向量
    def rule_lift(self):
        A = self.points[0]
        B = self.points[-1]
        if B.x > A.x and B.y > A.y:
            return True
        else:
            return False
    # 撇点   ABC  3点   B 为fork点    A 向左下动到B向右下移动到C
    def rule_neglect_point(self):

        return True
    # 竖提   A B C
    def rule_vertical_lift(self):

        return True
    # 横竖提  A B C D
    def rule_across_vertical_lift(self):

        return True

    # 弯钩
    def rule_hook(self):

        return True
    # 竖钩
    def rule_vertical_hook(self):

        return True
    # 竖弯钩
    def rule_vertical_hook(self):

        return True
    # 斜钩
    def rule_slope_hook(self):

        return True
    # 卧钩
    def rule_lie_hook(self):

        return True
    # 横钩
    def rule_across_hook(self):

        return True
    # 横竖勾
    def rule_across_vertical_hook(self):

        return True
    # 横竖弯钩
    def rule_across_neglect_line_hook(self):

        return True
    # 横折折折钩
    def rule_across_change_change_change_hook(self):

        return True
    # 竖折折钩
    def rule_vertical_change_change_hook(self):

        return True
    # 竖弯
    def rule_vertical_bend_line(self):

        return True
    # 竖折弯
    def rule_across_change_bend_line(self):

        return True
    # 横折
    def rule_vertical_change(self):

        return True
    # 竖折
    def rule_vertical_change(self):

        return True
    # 撇折
    def rule_neglect_change(self):

        return True
    # 横撇
    def rule_across_neglect(self):

        return True
    # 横折折撇
    def rule_across_change_change_neglect(self):

        return True
    # 竖折撇
    def rule_vertical_change_neglect(self):

        return True