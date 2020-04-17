import numpy as np
import os
import matplotlib.pyplot as plt
# import sys
# import
import train
import model
import utils
import gmm
import tensorflow as tf
import random

from model import FontRNN, get_default_hparams, copy_hparams

# a = [[  28. , 6.  , 0.],
# [  103.,  10. ,  1.]]


# print(a[0:1 + 1,0])

plt.plot([0.,6.], [20.,156.], color='black',  linewidth=3)
plt.show()

a = '123456789'
b = a[:-1]
print(b)