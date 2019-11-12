'''
Python 3.6
Pytorch 0.4
Written by Hongyu Wang in Beihang university
'''
import os
import cv2
import sys
# import pandas as pd
import pickle as pkl
import numpy
from scipy.misc import imread, imresize, imsave
#
# image_path='/Users/momo/Desktop/songyang/smallschool/数学公式数据的采集toMM-1012/数学公式数据的采集-1/'
# outFile='/Users/momo/Desktop/songyang/smallschool/数学公式数据的采集toMM-1012/offline-train.pkl'
# oupFp_feature=open(outFile,'wb')
#
# features={}
#
# channels=1
#
# sentNum=0
#
# # scpFile=open('/Users/momo/Desktop/math_formular_data/a3801-3918.txt',encoding="utf8")
# scpFile=open('/Users/momo/Desktop/songyang/smallschool/数学公式数据的采集toMM-1012/1131-1280.txt')
# while 1:
#     line=scpFile.readline().strip() # remove the '\r\n'
#     print(line)
#     if not line:
#         break
#     else:
#         key = line.split()[0]
#         print(key)
#         image_file = image_path + key  + '.png'
#         im = imread(image_file)
#         mat = numpy.zeros([channels, im.shape[0], im.shape[1]])
#         for channel in range(channels):
#             image_file = image_path + key + '.png'
#             im = imread(image_file)
#             mat[channel,:,:] = im
#         sentNum = sentNum + 1
#         features[key] = mat
#         if sentNum / 500 == sentNum * 1.0 / 500:
#             print('process sentences ', sentNum)
#
# print('load images done. sentence number ',sentNum)
#
# pkl.dump(features,oupFp_feature)
# print('save file done')
# oupFp_feature.close()
dir_path ='/Users/momo/Desktop/huizhen/样本不平衡/choose_ideal/'
save_path ='/Users/momo/Desktop/huizhen/样本不平衡/choose_ideal_1/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for name in os.listdir(dir_path):
    if not name.endswith('.png') : continue

    img =cv2.imread(dir_path+name)
    ext = os.path.splitext(name)[0]
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # _, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    # cv2.imshow("img",img)
    # cv2.waitKey()
    cv2.imwrite(save_path+ext+'.bmp',img)


