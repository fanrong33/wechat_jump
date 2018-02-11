# encoding: utf-8
# 用于生成plan2神经网络的训练数据
# 在train_data文件夹会生成以数字顺序命名的图片
# time.npz 文件用数组方式保存了每个图片对应的按压时间（毫秒）

import time
from PIL import Image
import numpy as np
import os



def save_train_data(press_time):
    """
    运行过程一边存储训练图片
    images.npz  => 对应'时间戳' 1518092603.jpg 
    labels.npz  => 按压时间（毫秒）
    build，随着运行的不断越来越完善
    """
    img = Image.open('autojump.png')
    # 将图片压缩，并截取中间部分，截取后为100*100
    img = img.resize((108, 192), Image.ANTIALIAS)
    region = (4, 50, 104, 150)
    img = img.crop(region)
    # 转化为jpg
    bg = Image.new("RGB", img.size, (255,255,255))
    bg.paste(img)
    bg = bg.resize((56, 56), Image.ANTIALIAS)
    timestamp = int(time.time())
    file_name = str(timestamp) + ".jpg"
    bg.save("./train_data1/"+file_name)


    with open('./train_data1/train.txt', 'a') as fp:
        fp.write(' {}  {} \n'.format(timestamp, press_time))

    print("build train data {%s :%s}" % (timestamp, press_time))

def resize_img():
    files = os.listdir('./train_data1/')
    for i in files:
        if i[-3:] == 'jpg':
            img = Image.open('./train_data1/'+i)
            # 将图片压缩，并截取中间部分，截取后为100*100
            img = img.resize((56, 56), Image.ANTIALIAS)
            img.save('./train_data1/'+i)


def show_dict():
    xy = np.loadtxt('./train_data1/train.txt', unpack=True, dtype='int')
    # print(xy)
    # print(xy[0])
    # print(xy[1])


if __name__ == '__main__':
    # save_train_data(300)
    show_dict()
    # resize_img()
