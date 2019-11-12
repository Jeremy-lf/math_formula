import os
import cv2
import re
import random


#补边并查找异常
#
file = open('/Users/momo/Desktop/huizhen/初中公式提交-10.18/value_normal_merge.txt')


dir_pic ='/Users/momo/Desktop/huizhen/初中公式提交-10.18/'
save_pic = '/Users/momo/Desktop/huizhen/padding/'
if not os.path.exists(save_pic):
    os.makedirs(save_pic)
i =0
while 1:
    line=file.readline().strip() # remove the '\r\n'
    # print(line)
    if not line:
        break
    else:
        key = line.split()[0]
        # print(key)
        path = dir_pic+key+'.png'
        if os.path.exists(path):
            a = random.randint(4, 10)
            b = random.randint(4, 10)
            c = random.randint(4, 10)
            d = random.randint(4, 10)
            img = cv2.imread(path)
            img = cv2.copyMakeBorder(img, a, b, c, d, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.imwrite(save_pic+key+'.png',img)
        else:
            print(path)
        i = i+1
print(i)




#找出两个列表中的重复的元素


# file = open('/Users/momo/Desktop/huizhen/初中公式提交-10.18/num.txt')
# file_1 = open('/Users/momo/Desktop/huizhen/初中公式提交-10.18/num_1.txt')
# f = file.readlines()
# f1 = file_1.readlines()
#
# set1 = set(f)
# set2 = set(f1)
#
# print(set1&set2)
# print(set1^set2)



#一个列表中的重复的元素

# import collections
#
# file = open('/Users/momo/Desktop/huizhen/初中公式提交-10.18/num.txt')
#
# f = file.readlines()
#
#
# print([item for item, count in collections.Counter(f).items() if count > 1])

