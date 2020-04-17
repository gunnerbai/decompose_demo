import numpy as np
import cv2
import matplotlib as plt
import fontforge
def threshold_image(image, min_p = 200, max_p = 255):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayscaled, min_p, max_p, cv2.THRESH_BINARY)
    return threshold

# path = 'tupian1.png'
# img = cv2.imread(path)
#
# threshold = threshold_image(img)
# threshold = cv2.bitwise_not(threshold)
# threshold[threshold == 255] = 1

fontforge.open('demo.ttf')


# cv2.imshow('1',threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()