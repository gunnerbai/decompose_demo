import numpy as np
import os
import matplotlib.pyplot as plt
# import sys
import pycpd
import train
import model
import utils
import gmm
import tensorflow as tf
import random

from model import FontRNN, get_default_hparams, copy_hparams

def draw(delta_gt_stroke):
     ground_truth_stroke = delta_gt_stroke.copy()
     # stroke = delta_stroke.copy()
     # print('ground_truth_stroke1111', ground_truth_stroke[:, :])
     # print('ground_truth_stroke1111',type(ground_truth_stroke))
     # convert to absolute coordinate
     scale_factor = 1
     low_tri_matrix = np.tril(np.ones((delta_gt_stroke.shape[0], delta_gt_stroke.shape[0])), 0)
     ground_truth_stroke[:, :2] = np.rint(scale_factor * np.matmul(low_tri_matrix, delta_gt_stroke[:, :2]))

     # low_tri_matrix = np.tril(np.ones((delta_stroke.shape[0], delta_stroke.shape[0])), 0)
     # stroke[:, :2] = np.rint(scale_factor * np.matmul(low_tri_matrix, delta_stroke[:, :2]))

     plt.figure(figsize=(3, 3))
     # plt.subplot(121)
     # plt.xlim(0, 300)
     # plt.ylim(0, 300)
     pre_i = 0
     print('ground_truth_stroke:',ground_truth_stroke)
     print('ground_truth_stroke:',type(ground_truth_stroke))
     for i in range(ground_truth_stroke.shape[0]):
          if ground_truth_stroke[i][2] == 1:
              z1 = ground_truth_stroke[pre_i:i + 1, 0]
              z2 = ground_truth_stroke[pre_i:i + 1, 1]
              print('位置是:',i)
              print('z1:',z1)
              print('z2:',z2)
              plt.plot(z1,z2 , color='black',linewidth=1)
              pre_i = i + 1
     # plt.axis('off')
     plt.gca().invert_yaxis()
     # plt.title("Groud Truth")

     # plt.subplot(122)
     # plt.xlim(0, 300)
     # plt.ylim(0, 300)
     # pre_i = 0
     # for i in range(stroke.shape[0]):
     #      if stroke[i][2] == 1:
     #           plt.plot(stroke[pre_i:i + 1, 0], stroke[pre_i:i + 1, 1], color='black', linewidth=3)
     #           pre_i = i + 1
     # plt.axis('off')
     # plt.gca().invert_yaxis()
     # plt.title("Generated")
     # plt.imsave()
     plt.savefig('tessstttyyy.png', dpi=1000)
     plt.show()

# data_filepath = '../data/FZTLJW_775.npz'
data_filepath1 = '../data/arr.npy'
# data = np.load(data_filepath, allow_pickle=True, encoding='latin1')
data = np.load(data_filepath1)
# data = np.load('../data/FZTLJW_775.npz')
print(data[0])
# x_train = data['train']
# y_test = data['test']
# y_std_test = data['std_test']
# y_std_train = data['std_train']
# valid = data['valid']

# print(len(x_train))
# print(x_train[7])
# print('x_train',type(x_train))
# draw(x_train[7])
# draw(y_std_train[0])
# for i in data:
#     draw(i)
draw(data)
# font = y_test[2]
# font1 = x_train[3]
# fong_1  = valid[1]
# fong_2  = y_std_test[2]
# fong_3  = y_std_train[3]

#
# print('fong_3[1]:',font)
# print('font1[1]:',fong_2)
# a = font
# a = font
# b = fong_2
# draw(a)
# draw(b)
# print(len(y_train))
# for i in range(len(x_train)):
    # im = Image.fromarray(x_train[i])
    # im.show()
# im = Image.fromarray(x_train[0])
# print(im.shape)
# im.show()