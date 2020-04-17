import numpy as np
import cv2


class Strokes_type(object):



# ● 橫：Horizontal（代表字母：H）。
#  ● 豎：Vertical（代表字母：V）。
#  ● 撇：Throw（代表字母：T）。
#  ● 捺：Press（代表字母：P）。
#  ● 點：Dot（代表字母：D）。
#  ● 挑：Upward horizontal（代表字母：U）。
#  ● 彎：Clockwise curve（代表字母：C）。
#  ● 曲：Anticlockwise curve（代表字母：A）。
#  ● 鈎：J-hook（代表字母：J）。
#  ● 圈：Oval（代表字母：O）。
# ◆ 變化形態筆畫：Deformed stroke
#  ● 扁：Flat（代表字母：F）。
#  ● 直：Wilted（代表字母：W）。
#  ● 斜：Slanted（代表字母：S）。
#  ● 左：Left（代表字母：L）。
#  ● 右：Right（代表字母：R）。
#  ● 長：Extended（代表字母：E）。
#  ● 短：Narrowed（代表字母：N）。
#  ● 倒：Inverted（代表字母：I）。
#  ● 反：Mirrored（代表字母：M）。
    def __init__(self,points,fork_list):
            self.points     =   points




