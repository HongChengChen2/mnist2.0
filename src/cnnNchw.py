# coding=utf-8
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tempfile
import numpy as np
import tensorflow as tf
import argparse
from os import environ
from models.utils import *

FLAGS = None

#tensorflow的官方例子：卷积->池化->卷积->池化->全连接->dropout->全连接->训练->测试

def deepnn(x):
    with tf.name_scope('reshape'):
        # -1表示由输入的图像决定 ，样本数量/宽度/高度/颜色通道数
        x_image0 = tf.reshape(x, [-1, 28, 28, 1])

        # 自带的是nhwc，转化为适应GPU的nchw
        x_image = tf.transpose(x_image0, [0, 3, 1, 2])

    ## 第一层卷积操作 ##
    with tf.name_scope('conv1'):

        #定义一个tensor来保存shape为[5,5,1,32]权重矩阵W：前两个参数是窗口的大小，
        # 第三个参数是channel的数量。最后一个定义了我们想使用多少个特征。更进一步，还需要为每一个权重矩阵定义bias
        W_conv1 = weight_variable([5, 5, 1, 32])

        b_conv1 = bias_variable([32])
        conv = conv2d(x_image, W_conv1)
        conv = tf.nn.bias_add(conv, b_conv1, data_format='NCHW')
        print(conv.get_shape().as_list())

        h_conv1 = tf.nn.relu(conv)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
        print(h_pool1.get_shape().as_list())

    # Second convolutional layer -- maps 32 feature maps to 64.
    ## 第二层卷积操作 ##
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2_0 =conv2d(h_pool1, W_conv2)
        h_conv2_0 =tf.nn.bias_add(h_conv2_0, b_conv2, data_format='NCHW')

        h_conv2 = tf.nn.relu( h_conv2_0 )

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    ## 第三层全连接操作 ##
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #drop防止过拟合
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## 第四层输出操作 ##
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 90])
        b_fc2 = bias_variable([90])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")


def max_pool_2x2(x):
    '''return tf.layers.max_pooling2d(x, pool_size=[2,2], strides=[1, 2, 2, 1],
          padding='valid', data_format='NCHW'
      )'''
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 2],
                          strides=[1, 1, 2, 2], padding='SAME', data_format="NCHW")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main(_):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    save_path = '../data/preprocessed_data/mnist_100'
    '''
    train_labels = load_pkls(save_path, 'train_labels') # 数据类型是 ndarray
    train_imgs = load_pkls(save_path, 'train_images')
    test_labels = load_pkls(save_path, 'test_labels')
    test_imgs = load_pkls(save_path, 'test_images')
    
    '''
    train_imgs = load_pkls(save_path, 'x_train')
    train_labels = load_pkls(save_path, 'y_train')
    valid_imgs =  load_pkls(save_path, 'x_valid')
    valid_labels =  load_pkls(save_path, 'y_valid')
    test_imgs =  load_pkls(save_path, 'x_test')
    test_labels =  load_pkls(save_path, 'y_test')

    # Create the model
    # 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
    x = tf.placeholder(tf.float32, [None, 28,28,1])

    # 类别y是10-99总共90个类别，对应输出分类结果
    y_ = tf.placeholder(tf.float32, [None, 90])

    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000): #训练图片的数量
            this_imgs = train_imgs[:50]
            train_imgs = train_imgs[50:]
            this_labels = train_labels[:50]
            train_labels = train_labels[50:]

            if i % 1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: this_imgs, y_: this_labels, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: this_imgs, y_: this_labels, keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_imgs, y_: test_labels, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')

    #添加下面3句，使可以在GPU运行,还需要import argparse 、from os import environ
    '''parser.add_argument('-g', '--gpu', nargs=1,
                        choices=[0, 1], type=int, metavar='',
                        help="Run single-gpu version."
                             "Choose the GPU from: {!s}".format([0, 1]))
    args = parser.parse_args()
    if args.gpu:
        environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu[0])
        mode_ = 'single-gpu'''

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)