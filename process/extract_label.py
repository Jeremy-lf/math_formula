

# file = open('/Users/momo/Desktop/songyang/data_3/sum_4.txt')

file_1 = open('/Users/momo/Desktop/songyang/data_3/num.txt')
file_2 = open('/Users/momo/Desktop/songyang/data_3/sum_4_norm.txt')
# file_3 = open('/Users/momo/Desktop/songyang/highschool/1_merge/value_normal.txt','w')

file_4 = open('/Users/momo/Desktop/songyang/data_3/sum_4_normal_merge.txt','w')
# while 1:
#     line=file.readline().strip() # remove the '\r\n'    #c2001 - c3981
#     print(line)
#
#     if not line:
#         break
#     else:
#         key = line.split()[0]
#         value = line.split(' ',1)[1]
#         # print(key)
#         print(value)
#         file_1.write(str(key)+'\n')
#         file_2.write(str(value)+'\n')



#1538
# 合并索引和标签
#
while 1:
    line_1 = file_1.readline().strip() # remove the '\r\n'
    line_2 = file_2.readline().strip()

    if not line_1:
        break
    else:
        file_4.write(str(line_1)+' '+str(line_2)+'\n')

