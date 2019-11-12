


from collections import Counter

#######计算词频#####

#
# test_label ='/Users/momo/Desktop/songyang/highschool/1_merge/label_2_no_sin_font/test_label_no_sin_font.txt'
#
# test_num_label ='/Users/momo/Desktop/songyang/highschool/1_merge/label_2_no_sin_font/test_word_num.txt'


# train_label = '/Users/momo/Desktop/huizhen/version_6/pre_train_wrong_label.txt'
#
# train_num_label ='/Users/momo/Desktop/huizhen/version_6/pre_train_num_.txt'
#
#
#
# f = open(train_label)
# f1 = open(train_num_label,'a')
#
# word_list=[]
# word_dict={}
#
#
# with open(train_label) as f:
#
#     # line = f.readline().split()
#     lines = f.readlines()
#     for i , line in enumerate(lines):
#         line=line.strip().split()[1:]
#         print("num:",i,"line:",line)
#         #print("sentence;",f.readline().split()[0])
#         for word in line:
#             if word not in word_dict:
#                 word_dict[word] = 1
#             else:
#                 word_dict[word] += 1
#
#
#
# items = list(word_dict.items())
# items.sort(key=lambda x:x[1], reverse=True)
# print(items)
#
# word_dict = dict(items)
#
# for key in word_dict:
#     f1.write(key +'\t'+ str(word_dict[key])+'\n')
#
# print(len(word_dict.keys()))




train_label= '/Users/momo/Desktop/huizhen/pre_train_wrong_label.txt'

split_file = open('/Users/momo/Desktop/huizhen/split_pred_label.txt','a')
split_file_1 = open('/Users/momo/Desktop/huizhen/split_real_label.txt','a')
with open(train_label) as f:

    # line = f.readline().split()
    lines = f.readlines()
    for i , line in enumerate(lines):
        line=line.strip().split('\t',1)
        print("num:",i,"line:",line)
        split_file.write(line[0]+'\n')
        split_file_1.write(line[1]+'\n')
        #print("sentence;",f.readline().split()[0])


