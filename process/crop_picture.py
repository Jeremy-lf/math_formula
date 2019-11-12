import cv2
import os
from PIL import Image
import numpy

for i in range(11,12):

    dir_path ='/Users/momo/Desktop/songyang/num_3/中小学公式采集_%d/' % i
    save_path = '/Users/momo/Desktop/songyang/num_3/crop_%d/' % i
    save_path_1 ='/Users/momo/Desktop/songyang/num_3/crop_padding_%d/' % i
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_1):
        os.makedirs(save_path_1)
    for name in os.listdir(dir_path):
        if not name.endswith('.png') : continue

        ext = os.path.splitext(name)[0]

        im_1 = Image.open(dir_path + ext+'.png')
        # im_1 = cv2.imread(dir_path + ext+'.png')
        # cv2.imshow("img",im)
        # cv2.waitKey()
        w ,h = im_1.size
        im_1 = numpy.asarray(im_1)
        h1,w1 = im_1.shape[:2]
        print(h1,w1)
        # cv2.imshow("img",im_1)
        # cv2.waitKey()
        if 235<=h1:
            im_2 = im_1[40:h1-40,40:w1-40]
        if 150<h1<235:
            im_2 = im_1[35:h1-35,35:w1-35]
        if 150>=h1>100:
            im_2 = im_1[22:h1-22,23:w1-23]
        if 100>=h1>=80:
            im_2 = im_1[11:h1 - 11, 12:w1 - 12]
        if 80>h1>=60:
            im_2 = im_1[4:h1 - 4, 4:w1 - 4]
        if 60>h1>45:
            im_2 = im_1[2:h1 - 2, 2:w1 - 2]
        if  h <= 45:
            im_2 = im_1
        print(im_2.shape[:2])
        cv2.imwrite(save_path+ext+'.png',im_2)
        #im_2 = Image.open(save_path+ext+'.png')


        try:
            im_2 = Image.open(save_path+ext+'.png')
            x,y = im_2.size
            try:
                # 使用白色来填充背景 from：www.outofmemory.cn
                # (alpha band as paste mask).
                p = Image.new('RGBA', im_2.size, (255,255,255))
                p.paste(im_2, (0, 0, x, y), im_2)
                p.save(save_path_1+ext+'.png')
            except:
                print(ext)
        except:
            print(name)




