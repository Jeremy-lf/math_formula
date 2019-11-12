
import os
from PIL import Image

# for i in range(1,9):
i = 2
dir_path ='/Users/momo/Desktop/songyang/highschool/2/中小学公式采集-%d/' % i
save_path = '/Users/momo/Desktop/songyang/highschool/resize_2/formula_images_%d/' % i
for name in os.listdir(dir_path):
    if not name.endswith('.png') : continue
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ext = os.path.splitext(name)[0]
    try:
        im = Image.open(dir_path+name)
        x,y = im.size
        try:
            # 使用白色来填充背景 from：www.outofmemory.cn
            # (alpha band as paste mask).
            p = Image.new('RGBA', im.size, (255,255,255))
            p.paste(im, (0, 0, x, y), im)
            p.save(save_path+ext+'.png')
        except:
            print(ext)
    except:
        print(name)

