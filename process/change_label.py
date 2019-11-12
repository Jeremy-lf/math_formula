import os
import cv2
import re


# 修改标签

# c =21000
# file = open('/Users/momo/Desktop/generator_latex.txt')
#
# file_1 = open('/Users/momo/Desktop/generator_latex_1.txt','w')
# while 1:
#     line=file.readline().strip() # remove the '\r\n'
#     print(line)
#     if not line:
#         break
#     else:
#         key = line.split()[0]
#         value = line.split(' ',1)[1]
#         # print(key)
#         print(value)
#         file_1.write(str(c) +' '+str(value)+'\n')
#         print(c)
#         # file_1.write(line+'\n')
#         c = c+1






# # i =1
#修改图片名字
#
i =11

dir_path ='/Users/momo/Desktop/q_1/'
save_path = '/Users/momo/Desktop/q_2/'
e=21000
directory = os.listdir(dir_path)
if '.DS_Store' in directory:
    directory.remove('.DS_Store')
# directory.sort()

#directory.sort(key=lambda x: int(re.match('\D+(\d+)\.png', x).group(1)))

directory.sort(key=lambda x: int(re.match('(\d+)\.bmp', x).group(1)))

print(directory)
for name in directory:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not name.endswith('.bmp'): continue
    img = cv2.imread(dir_path+name)
    cv2.imwrite(save_path+str(e)+'.bmp',img)
    print(save_path+str(e)+'.bmp')
    e = e+1



#合并标签文件

# dir = '/Users/momo/Desktop/songyang/highschool/2_merge/'
# save = '/Users/momo/Desktop/songyang/highschool/2_merge/sum.txt'
#
#
# directory = os.listdir(dir)
# # directory.sort(key=lambda x: int(re.match('\D+(\d+)\.png', x.group(1))))#
# directory.sort()
#
# file_3 = open(save,'w')
#
#
# for name in directory:
#
#     dir_path = os.path.join(dir,name)
#     file_1 = open(dir_path)
#
#     while 1:
#         line_1 = file_1.readline().strip()  # remove the '\r\n
#
#         if not line_1:
#             break
#         else:
#             file_3.write(str(line_1) + '\n')




#合并图片文件


# save_path = '/Users/momo/Desktop/songyang/num_3/sum_4/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# for  i in range(1,12):
#
#     dir_path = '/Users/momo/Desktop/songyang/num_3/order_pic_%d/' % i
#     for name in os.listdir(dir_path):
#         if not name.endswith('.png'): continue
#         ext = os.path.splitext(name)[0]
#         img = cv2.imread(dir_path+name)
#         cv2.imwrite(save_path+ext+'.png',img)