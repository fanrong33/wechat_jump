# encoding: utf-8
# 转化成1维的数据
# 使用最基本的神级网络训练一个模型
# 先不保存模型
# 测试准确率
# version 1.0.2 build 20180211

'''
TensorFlow 1.0.0
'''

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import time

import screencap


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # VALID


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # VALID


def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME') # VALID


# 输入：56*56的灰度图片，前面的None是batch size, 这里都为1
x  = tf.placeholder(tf.float32, shape=[None, 56, 56, 1], name='x-input')
# 输出：一个浮点数，就是按压时间，单位为毫秒
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')


# 第一层卷积 32个feature map
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) # output size 56x56x32
h_pool1 = max_pool_2x2(h_conv1) # 池化后为28x28x32


# 第二层卷积 32个feature map
W_conv2 = weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 卷积后为 28*28*64
h_pool2 = max_pool_2x2(h_conv2) # 池化后为14*14*64


# 全连接层14*14*64 --> 32
W_fc1 = weight_variable([14*14*64, 32])
b_fc1 = bias_variable([32])
h_pool2_flat = tf.reshape(h_pool2, [-1, 14*14*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# drapout, 训练时为0.6，play时为1
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 32 --> 1
W_fc2 = weight_variable([32, 1])
b_fc2 = bias_variable([1])
y_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2




# 获取屏幕截图并转换为模型的输入
def get_screenshot():
    # 使用PIL处理图片，并转为jpg
    im = Image.open(r"./autojump.png")
    w, h = im.size
    # 将图片压缩，并截取中间部分，截取后为100*100
    im = im.resize((108, 192), Image.ANTIALIAS)
    region = (4, 50, 104, 150)
    im = im.crop(region)
    # 转换为jpg
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im)
    bg = bg.resize((56, 56), Image.ANTIALIAS)
    bg.save(r"./autojump.jpg")

    img = Image.open('./autojump.jpg').convert("L")
    # print(img.shape)
    ''' (100, 100) '''

    img_data = np.array(img).reshape(56, 56, 1)

    # 因为输入shape有batch维度，所以还要套一层
    return np.array([img_data/255.])


    # img_data = tf.image.decode_jpeg(tf.gfile.FastGFile('./autojump.jpg', 'rb').read())
    # # 使用TensorFlow转为只有1通道的灰度图
    # img_data_gray = tf.image.rgb_to_grayscale(img_data)
    # x_in = np.asarray(img_data_gray.eval(session=sess), dtype='float32')

    # # [0,255]转为[0,1]浮点
    # for i in range(len(x_in)):
    #     for j in range(len(x_in[i])):
    #         x_in[i][j][0] /= 255

    # # 因为输入shape有batch维度，所以还要套一层
    # return np.array([x_in])


def jump(press_time):
    """
    按压press_time时候后松开，完成一次跳跃
    """
    print('press time: %sms' % press_time)

    rand = random.randint(0, 9) * 10
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {time}'.format(x1=750+rand, y1=1500+rand, x2=750+rand, y2=1500+rand, time=press_time)
    os.system(cmd)
    print(cmd)

def has_die(x_in):
    """
    判断是否游戏失败到分数页面
    """
    # 判断左上右上左下右下四个点的亮度
    if (x_in[0][0][0][0] < 0.4) and \
        (x_in[0][0][len(x_in[0][0]) - 1][0] < 0.4) and \
        (x_in[0][len(x_in[0]) - 1][0][0] < 0.4) and \
        (x_in[0][len(x_in[0]) - 1][len(x_in[0][0]) - 1][0] < 0.4):
        return True
    else:
        return False


def restart():
    """
    游戏失败后重新开始，(540，1588)为1080*1920分辨率手机上重新开始按钮的位置
    """
    cmd = 'adb shell input swipe 540 1588 540 1588 10'
    os.system(cmd)
    time.sleep(1)


def start_play():
    """
    开始游戏
    """
    # 模型持久化
    saver = tf.train.Saver()

    # 启动Graph
    with tf.Session() as sess:

        model_file = tf.train.latest_checkpoint('save/')
        if model_file:
            # 加载模型
            saver.restore(sess, model_file)
        else:
            print('No checkpoint file found')
            return

        # screencap.check_screencap()
        while True:
            # screencap.pull_screencap()

            x_input = get_screenshot()
            # print(x_input.shape)
            if has_die(x_input):
                print('game over')
                restart()
                continue

            # 神经网络预测的输出
            y_pred = sess.run(y_fc2, feed_dict={x: x_input, keep_prob: 1})
            print(y_pred)

            touch_time = int(y_pred[0][0]*1000)
            jump(touch_time)

            time.sleep(random.uniform(1.2, 1.5))


if __name__ == '__main__':
    start_play()


