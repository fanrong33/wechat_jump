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


# 准备数据集
xy = np.loadtxt('train_data2/train.txt', unpack=True, dtype='int')
X = []
images = xy[0]
for i in images:
    filename = "./train_data2/%d.jpg" % i
    img = np.array(Image.open(filename).convert("L"))
    # print(img.shape)
    ''' (100, 100) '''

    img_data = np.array(img).reshape(56, 56, 1)
    X.append(img_data/255.)
    
    
    # 方法三， 使用tf的图片处理接口，速度非常慢非常慢
    # img_data = tf.image.decode_jpeg(tf.gfile.FastGFile(filename, 'rb').read())
    # 使用TensorFlow转为只有1通道的灰度图
    # img_data_gray = tf.image.rgb_to_grayscale(img_data)

    # x_in = np.asarray(img_data_gray.eval(session=sess), dtype='float32')

    # [0,255]转为[0,1]浮点
    # for i in range(len(x_in)):
    #     for j in range(len(x_in[i])):
    #         x_in[i][j][0] /= 255
    # X.append(x_in)

X = np.array(X)
# print(X.shape)
''' (347, 100, 100, 1) '''
'''
[[ 0.88235294  0.88235294  0.88235294 ...,  0.83137255  0.83137255
   0.83137255]
 ...,
 [ 0.88235294  0.88235294  0.88235294 ...,  0.83137255  0.83137255
   0.83137255]]
'''

Y = []
labels = xy[1]
for i in labels:
    Y.append([float(i/1000.)])
# print(Y)
'''
[[713], [558], ..., [733], [542]]
'''

dataset_size = len(X)

# 定义训练数据 batch 的大小
batch_size = 1


# 定义计算精度函数
def compute_accuracy(v_xs, v_ys):
    global y_fc2
    y_pred = sess.run(y_fc2, feed_dict={x: v_xs})

    # 预测的y_与实际数据集的y进行对比判断是否相等
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))
    # 统计正确的比例
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys})
    return result


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


# 输入：56*56的灰度图片，前面的None是batch size, 这里长度都为1
x  = tf.placeholder(tf.float32, shape=[None, 56, 56, 1], name='x-input')
# 输出：一个浮点数，就是按压时间，单位为s
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')


# 第一层卷积 32个feature map
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) # 卷积后为 56x56x32
h_pool1 = max_pool_2x2(h_conv1) # 池化后为 28x28x32


# 第二层卷积 32个feature map
W_conv2 = weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 卷积后为 28x28x64
h_pool2 = max_pool_2x2(h_conv2) # 池化后为 14x14x64


# 全连接层 14x14x64 --> 32
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


# 学习率
learn_rate = tf.placeholder(tf.float32, name='learning_rate')

# 因输出直接是时间值，而不是分类概率，所以用平方损失函数
cost = tf.reduce_mean(tf.square(y_fc2 - y_))
train = tf.train.AdamOptimizer(learn_rate).minimize(cost)


# 模型持久化
saver = tf.train.Saver()


# 启动Graph
with tf.Session() as sess:

    model_file = tf.train.latest_checkpoint('save/')
    if model_file:
        # 加载模型
        saver.restore(sess, model_file)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)


    # 设定训练的轮数
    for step in range(500001):

        # 每次选取batch_size个样本进行训练
        start = (step * batch_size) % dataset_size
        end   = min(start+batch_size, dataset_size)
        
        batch_xs, batch_ys = X[start:end], Y[start:end]
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.6, learn_rate: 0.00002})


        y_pred = sess.run(y_fc2, feed_dict={x: batch_xs, keep_prob: 1})
        # loss 计算损失
        loss = sess.run(cost, feed_dict={y_fc2: y_pred, y_: batch_ys})
        print("After %s training step(s), y_: %s, y_pred: %s, loss is %s" % (step, batch_ys, y_pred, loss))


        # total_accuracy = compute_accuracy(X, Y)
        # print("After %d training step(s), total accuracy is %g, loss is %s" % (step, total_accuracy, loss))

        if step % 5000 == 0:

            # 持久化模型
            saver.save(sess, 'save/model.mod')





