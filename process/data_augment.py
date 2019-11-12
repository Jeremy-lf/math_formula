import numpy as np
import os
import cv2
import copy
import random
import glob

# def warp(img):
#
#     _,h,w =img.shape
#     maxs= max(h,w)
#     dst_size = maxs
#     src_middle = np.array([h/2, w/2])
#     dst_middle = np.array([0.5, 0.5])
#     gesture_scale = random.choice([1.2])
#
#     # gesture_scale = 0.9
#     rotate_degree = 0
#
#     # if data_aug = True:
#     #     dst_middle += np.array([(random.random() - 0.5) * 2 * 0.25,
#     #                             (random.random() - 0.5) * 2 * 0.25])
#     #     gesture_scale += (random.random() - 0.5) * 2 * 0.4
#     #     rotate_degree += (random.random() - 0.5) * 2 * 12
#
#     dst_middle = dst_middle * dst_size
#     scale = dst_size / maxs * gesture_scale
#     offset = dst_middle - src_middle
#
#     M = cv2.getRotationMatrix2D((src_middle[0], src_middle[1]), rotate_degree, scale)
#     M[:, 2] += offset
#     img = cv2.warpAffine(img, M, (dst_size, dst_size))  # 仿射变换
#     return img
#
#
# def resize(image,  inter=cv2.INTER_AREA):
#     # 初始化缩放比例，并获取图像尺寸
#
#     h, w, _ = image.shape
#     print(h,w)
#     h2 =h *random.choice([0.8,1.1,1.2,0.9])
#     ratio = h2 / h
#     if h2>h:
#         image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
#     else:
#         image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
#     print("process:",(image.shape[:]))
#     # h2, w2, _ = image.shape
#     # if w2 > 240:
#     #     image = cv2.resize(image, (0, 0), fx=320 / w2, fy=32 / h2, interpolation=cv2.INTER_CUBIC)
#
#     return image
#
#
#
# def blur(img):
#     max_size = int(min(img.shape[0], img.shape[1]) * 0.05)
#     if max_size == 0:
#         max_size = 3
#     kernel_size = np.random.randint(max_size)
#     if kernel_size % 2 == 0:
#         kernel_size = kernel_size + 1
#         img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)  # 图像滤波，模糊程度
#     return img
#
#
#
# if __name__=="__main__":
#     # demo=DataAugment(debug=True)
#     image=cv2.imread("/Users/momo/Desktop/huizhen/train_data/90.bmp")
#     # outimg=warp(image)
#     # outimg = resize(image)
#     outimg = blur(image)
#     cv2.imshow("img",outimg)
#     cv2.waitKey(5000)

    #
    # height = image.shape[0]
    # width = image.shape[1]
    # channels = image.shape[2]
    # pixel_data = np.array(image, dtype=np.uint8)
    # print(pixel_data)
    # for c in range(channels):
    #     for row in range(height):
    #         for col in range(width):
    #             level = pixel_data[row, col, c]
    #             pixel_data[row, col, c] = 255 - level
i=0
path = '/Users/momo/Desktop/songyang/highschool/1_merge/test_data_1/'
save_path = '/Users/momo/Desktop/songyang/highschool/1_merge/test_data_reverse/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for name in os.listdir(path):
    if not name.endswith('.bmp'):continue
    image=cv2.imread(path+name)
    ext = os.path.splitext(name)[0]
    img = 255-image

    cv2.imwrite(save_path+ext+'.bmp',img)
    print("process image",save_path+ext+'.bmp')
    i = i+1
#image = cv2.imread('/Users/momo/Downloads/Pytorch-Handwritten-Mathematical-Expression-Recognition-master/off_image_test/off_image_test/18_em_5_0.bmp')
# print(image.shape)
# img = np.array(image)
# img = img / 255.0
# print(img)
# img[:][:]=1
# print(img)
# cv2.imshow("img",img)
# cv2.waitKey()
