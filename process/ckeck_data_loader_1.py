


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
    batch_size=6
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
        def __init__(self,train,train_label,batch_size):
            self.train = train
            self.train_label = train_label
            self.batch_size = batch_size

        def __getitem__(self, index):
            # image = (numpy.array(self.train[index])[0]).transpose((1, 2, 0))
            # print(image.shape)
            # cv2.imshow("image",image)
            # cv2.waitKey(0)
            train_setting = torch.from_numpy(numpy.array(self.train[index]))
            label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)

            size = train_setting.size()
            train_setting = train_setting.view(1,size[2],size[3])
            label_setting = label_setting.view(-1)
            return train_setting,label_setting

        def __len__(self):
            return len(self.train)



    off_image_train = custom_dset(train,train_label,batch_size)
    off_image_test = custom_dset(test,test_label,batch_size)

    # collate_fn is writting for padding imgs in batch.
    # As images in my dataset are different size, so the padding is necessary.
    # Padding images to the max image size in a mini-batch and cat a mask.
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        img, label = zip(*batch)
        aa1 = 0
        bb1 = 0
        k = 0
        k1 = 0
        max_len = len(label[0])+1
        for j in range(len(img)):
            size = img[j].size()
            if size[1] > aa1:
                aa1 = size[1]
            if size[2] > bb1:
                bb1 = size[2]

        for ii in img:
            ii = ii.float()
            img_size_h = ii.size()[1]
            img_size_w = ii.size()[2]
            img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
            img_mask_sub_s = img_mask_sub_s*255.0

            # back_groud ='/Users/momo/Desktop/1.jpg'
            # image_bg = cv2.imread(back_groud)
            # image_bg = image_bg[0:aa1,0:bb1]
            #
            # image_bg = cv2.cvtColor(image_bg, cv2.COLOR_BGR2GRAY)

            # image_bg = torch.from_numpy(image_bg.transpose(2,0,1)).float()

            # image_bg = image_bg[0].unsqueeze(0)

            # image_bg = torch.from_numpy(image_bg[numpy.newaxis, :, :]).float()



            # img_size_h = ii.size()[1]
            # img_size_w = ii.size()[2]
            #
            #
            # ii = ii.numpy().transpose(1, 2, 0)
            #
            # img_ii_copy = image_bg.copy()[:, :, numpy.newaxis]
            # img_ii_copy[0:img_size_h, 0:img_size_w] = ii
            #
            # # cv2.imshow('img_ii_copy', img_ii_copy)
            # # cv2.waitKey()
            # img_mask_sub_padding = torch.from_numpy(img_ii_copy.transpose(2,0,1))



            img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)   #1*2*h*w
            img_mask_sub = img_mask_sub.numpy().transpose(1, 2, 0)
            cv2.imshow('img_mask_sub', img_mask_sub)
            cv2.waitKey()
            padding_h = aa1-img_size_h
            padding_w = bb1-img_size_w
            #使用0填充输入tensor的边界(left,right,top,bottom)
            m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
            #对增加出来的通道进行补零

            img_mask_sub_padding = m(img_mask_sub)
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





    for i in range(5):

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














