


import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
from data_iterator import dataIterator
from Densenet_torchvision import densenet121
from Attention_RNN import AttnDecoderRNN
#from Resnet101 import resnet101
import random
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import sys
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import glob







def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])
    print('total words/phones',len(lexicon))
    return lexicon


if __name__=='__main__':






    batch_Imagesize=500000
    valid_batch_Imagesize=500000
    # batch_size for training and testing
    batch_size=12
    batch_size_t=6
    # the max (label length/Image size) in training a nd testing
    # you can change 'maxlen','maxImagesize' by the size of your GPU
    maxlen=24
    maxImagesize= 11000   #110000
    # hidden_size in RNN
    hidden_size = 256
    # teacher_forcing_ratio
    teacher_forcing_ratio = 1
    # change the gpu id
    gpu = [0]
    # learning rate
    lr_rate = 0.0001
    # flag to remember when to change the learning rate
    flag = 0
    # exprate
    exprate = 0





    datasets = ['/Users/momo/Desktop/songyang/highschool/1_merge/label_1/test_data_1.pkl', '/Users/momo/Desktop/songyang/highschool/1_merge/label_1/test_label_1.txt']
    valid_datasets = ['/Users/momo/Desktop/songyang/highschool/1_merge/label_1/test_data_1.pkl','/Users/momo/Desktop/songyang/highschool/1_merge/label_1/test_label_1.txt']
    dictionaries = ['/Users/momo/Desktop/dictionary_3.txt']

    # worddicts
    worddicts = load_dict(dictionaries[0])
    print(len(worddicts))
    worddicts_r = [None] * (len(worddicts) + 1)
    print(len(worddicts_r))
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk


    #load train data and test data
    train,train_label = dataIterator(
                                        datasets[0], datasets[1],worddicts,batch_size=1,
                                        batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
                                     )
    len_train = len(train)
    print("=========================", len_train)

    test,test_label = dataIterator(
                                        valid_datasets[0],valid_datasets[1],worddicts,batch_size=1,
                                        batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
                                    )
    len_test = len(test)


    class custom_dset(data.Dataset):
        def __init__(self,train,train_label,batch_size,is_train=True):
            self.train = train
            self.train_label = train_label
            self.batch_size = batch_size
            self.is_train = is_train

        def rotate_func(self, image):
            angle = random.choice([0.5, 0.6, 0.8, 0.9, 1.2, 1.4, 1.5])
            height, width = image.shape[:2]
            # angle = -45
            center = (width / 2, height / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

        def resize(self, img):

            h, w = img.shape[:2]
            h2 = h * random.choice([0.8, 1.1, 1.2, 0.9])
            ratio = h2 / h
            if h2 > h:
                image = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
            else:
                image = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

            return image

        def blur(self, img):
            max_size = int(min(img.shape[0], img.shape[1]) * 0.05)
            if max_size == 0:
                max_size = 3
            kernel_size = numpy.random.randint(max_size)
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            return img

        def __getitem__(self, index):
            # image = (numpy.array(self.train[index])[0]).transpose((1, 2, 0))
            # print(image.shape)
            # cv2.imshow("image",image)
            # cv2.waitKey(0)

            if self.is_train:
                img = numpy.array(self.train[index]).squeeze()
                # h, w = img.shape
                # img_1 = Image.fromarray(img)
                # img_2 = numpy.asarray(img)

                if random.random() < 0.5:
                    img = self.rotate_func(img)
                if random.random() < 0.5:
                    img = self.resize(img)
                if random.random() < 0.3:
                    img = self.blur(img)
                    # cv2.imshow("img",img_3)
                    # cv2.waitKey()
                h, w = img.shape
                train_setting = torch.from_numpy(numpy.array(img)).view(1, 1, h, w)
                label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(
                    torch.LongTensor)  # e=[[[1,2]],[[3,4]]]  -->np.array(e[0]) --->  array([[1, 2]]) ---> tensor([[1, 2]])

                size = train_setting.size()
                train_setting = train_setting.view(1, size[2], size[3])
                # print("========", train_setting.shape)
                label_setting = label_setting.view(-1)
                return train_setting, label_setting

            else:
                train_setting = torch.from_numpy(numpy.array(
                    self.train[index]))  # list->array,then array->tensor  numpy.array(self.train[index]) a picture

                # print('===========img', train_setting.shape)

                # train_setting = torch.from_numpy(numpy.array(self.train[index]))  #list->array,then array->tensor  numpy.array(self.train[index])
                label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(
                    torch.LongTensor)  # e=[[[1,2]],[[3,4]]]  -->np.array(e[0]) --->  array([[1, 2]]) ---> tensor([[1, 2]])

                size = train_setting.size()
                train_setting = train_setting.view(1, size[2], size[3])
                # print("========",train_setting.shape)
                label_setting = label_setting.view(-1)
                return train_setting, label_setting




            # train_setting = torch.from_numpy(numpy.array(self.train[index]))
            # label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)
            #
            # size = train_setting.size()
            # train_setting = train_setting.view(1,size[2],size[3])
            # label_setting = label_setting.view(-1)
            # return train_setting,label_setting

        def __len__(self):
            return len(self.train)



    off_image_train = custom_dset(train,train_label,batch_size,is_train=True)
    off_image_test = custom_dset(test,test_label,batch_size,is_train=False)

    # collate_fn is writting for padding imgs in batch.
    # As images in my dataset are different size, so the padding is necessary.
    # Padding images to the max image size in a mini-batch and cat a mask.
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        img, label = zip(*batch)
        aa1 = 0
        bb1 = 0
        aa1_1 = []
        bb1_1 = []
        k = 0
        k1 = 0
        max_len = len(label[0])+1
        for j in range(len(img)):
            size = img[j].size()
            aa1_1.append(size[1])
            bb1_1.append(size[2])

            # if size[1] > aa1:
            #
            #     aa1 = size[1]
            # if size[2] > bb1:
            #     bb1 = size[2]
        aa1_1.sort(reverse=True)
        bb1_1.sort(reverse=True)
        aa1 = aa1_1[0]
        bb1 = bb1_1[0]

        for ii in img:
            ii = ii.float()
            # img_size_h = ii.size()[1]
            # img_size_w = ii.size()[2]
            # img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
            # img_mask_sub_s = img_mask_sub_s*255.0
            f = glob.glob(r'/Users/momo/Desktop/数学公式识别/bg/*.png')
            print(f)
            back_groud = random.choice(f)
            #back_groud ='/Users/momo/Desktop/1.jpg'
            image_bg = cv2.imread(back_groud)
            image_bg = image_bg[0:aa1,0:bb1]

            image_bg = cv2.cvtColor(image_bg, cv2.COLOR_BGR2GRAY)

            # image_bg = torch.from_numpy(image_bg.transpose(2,0,1)).float()

            # image_bg = image_bg[0].unsqueeze(0)

            # image_bg = torch.from_numpy(image_bg[numpy.newaxis, :, :]).float()


            img_size_h = ii.size()[1]
            img_size_w = ii.size()[2]

            ii_1 = ii.numpy().transpose(1, 2, 0)
            print("ii shape:",ii.shape)
            if 1.6*img_size_h < aa1 and img_size_w*1.6 < bb1:

                ratio_h = img_size_h*1.25 / img_size_h
                ii_2 = cv2.resize(ii_1,(0, 0), fx=ratio_h, fy=ratio_h, interpolation=cv2.INTER_AREA)  #缩小图像INTER_AREA  放大图像INTER_CUBIC
            else:
                ii_2 = ii_1
                # else:
                #     ratio_w = bb1 / img_size_w
                #     ii = cv2.resize(ii, (0, 0), fx=ratio_w, fy=ratio_w, interpolation=cv2.INTER_AREA)

            img_ii_copy = image_bg.copy()[:, :, numpy.newaxis]

            img_size_h_1 = ii_2.shape[0]
            img_size_w_1 = ii_2.shape[1]

            h1 = int(aa1 - img_size_h_1)
            w1 = int(bb1 - img_size_w_1)

            if 0 <= h1 <= 6 :
                h1_random = random.randint(0, h1)
            elif 10>h1>6:
                h1_random = random.randint(0,h1-4)
            elif 10<=h1:
                h1_random = random.randint(0, h1 -9)
            # w1_random = random.randint(0,w1)
            if 4>=w1 >= 0:

                w1_random = random.randint(0,w1)
            elif 10>w1>=5:
                w1_random = random.randint(2, 5)
            elif w1>=10:
                w1_random = random.randint(0, 8)
            #
            # h1_random = random.randint(0, h1)
            # # w1_random = random.randint(0,w1)
            # if 2 >= w1 >= 0:
            #
            #     w1_random = random.randint(0, w1)
            # elif w1 > 2:
            #     w1_random = random.randint(0, 3)



            img_ii_copy[h1_random:h1_random+img_size_h, w1_random:w1_random+img_size_w] = ii_2


            #img_ii_copy[0:img_size_h, 0:img_size_w] = ii
            # cv2.imshow('img_ii_copy', img_ii_copy)
            # cv2.waitKey()

            img_mask_sub_padding = torch.from_numpy(img_ii_copy.transpose(2,0,1)).float()

            img_mask_sub_padding = torch.cat((img_mask_sub_padding,img_mask_sub_padding),dim=0)



            # img_ii = cv2.copyMakeBorder(ii, 0, padding_h, 0, padding_w,cv2.BORDER_CONSTANT,value=[255,255,255])
            # img_ii_tmp = torch.from_numpy(img_ii[numpy.newaxis, :, :])
            #
            # img_mask_sub_padding = torch.cat((img_ii_tmp, image_bg), dim=0)

            #img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)   #1*2*h*w
            # padding_h = aa1-img_size_h
            # padding_w = bb1-img_size_w
            #使用0填充输入tensor的边界(left,right,top,bottom)
            # m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
            #对增加出来的通道进行补零

            # img_mask_sub_padding = m(img_mask_sub)
            img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)

            if k==0:
                img_padding_mask = img_mask_sub_padding
            else:
                img_padding_mask = torch.cat((img_padding_mask,img_mask_sub_padding),dim=0)    #1*2*h*w---->6*2*h*w
            k = k+1

        for ii1 in label:
            ii1 = ii1.long()
            ii1 = ii1.unsqueeze(0)
            ii1_len = ii1.size()[1]
            m = torch.nn.ZeroPad2d((0,max_len-ii1_len,0,0))
            ii1_padding = m(ii1)
            if k1 == 0:
                label_padding = ii1_padding
            else:
                label_padding = torch.cat((label_padding,ii1_padding),dim=0)
            k1 = k1+1

        img_padding_mask = img_padding_mask/255.0
        return img_padding_mask, label_padding

    train_loader = torch.utils.data.DataLoader(
        dataset = off_image_train,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers=2,
        )


    test_loader = torch.utils.data.DataLoader(
        dataset = off_image_test,
        batch_size = batch_size_t,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers=2,
    )





    for i in range(100):

        for step, (batch_image, batch_label) in enumerate(train_loader):

            print(batch_image.shape)
            image_re = batch_image[0]
            print(image_re.shape)
            img_1 = batch_image[0][0, :, :].numpy()
            img_2 = batch_image[1][0, :, :].numpy()
            img_3 = batch_image[2][0, :, :].numpy()
            img_4 = batch_image[3][0, :, :].numpy()
            img_5 = batch_image[4][0, :, :].numpy()
            img_6 = batch_image[5][0, :, :].numpy()

            print('image size :', img_1.shape)
            cv2.imshow('img_1', img_1)
            cv2.imshow('img_2', img_2)
            cv2.imshow('img_3', img_3)
            cv2.imshow('img_4', img_4)
            cv2.imshow('img_5', img_5)
            cv2.imshow('img_6', img_6)
            # cv2.imshow('img_2', img_2)
            cv2.waitKey(0)














