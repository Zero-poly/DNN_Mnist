import tensorflow as tf
import numpy as np
from cnn_model_reload import reload_model
from cnn_model_reload import prediction
from PIL import Image


def load_mnist(path, kind='train'):
    import os
    import struct

    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8). \
            reshape(len(labels), 784)

    return images, labels


def weight(shape):
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init, name='W')


def bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name='b')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def variable_summary(var):
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))

    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)

    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)

    tf.summary.histogram('histogram', var)


def cnn(x, keep_prob):
    """
    简单的cnn分类mnist集
    :param x: (N,784)维的张量，784是每张图片的像素数目（28x28）
    :return: 元组(y,keep_prob).y是(N,10)维的张量，即训练结果
             keep_prob是一个标量占位符，表示随机丢弃的概率
    """
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('images', x_image, 10)

    with tf.name_scope('conv1'):
        with tf.name_scope('W'):
            W_conv1 = weight([5, 5, 1, 32])
            # variable_summary(W_conv1)
        with tf.name_scope('b'):
            b_conv1 = bias([32])
            # variable_summary(b_conv1)
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        with tf.name_scope('W'):
            W_conv2 = weight([5, 5, 32, 64])
            # variable_summary(W_conv2)
        with tf.name_scope('b'):
            b_conv2 = bias([64])
            # variable_summary(b_conv2)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('reshape'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    with tf.name_scope('fc1'):
        with tf.name_scope('W'):
            W_fc1 = weight([7 * 7 * 64, 1024])
            variable_summary(W_fc1)
        with tf.name_scope('b'):
            b_fc1 = bias([1024])
            variable_summary(b_fc1)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    with tf.name_scope('fc2'):
        with tf.name_scope('W'):
            W_fc2 = weight([1024, 10])
            variable_summary(W_fc2)
        with tf.name_scope('b'):
            b_fc2 = bias([10])
            variable_summary(b_fc2)

    with tf.name_scope('output'):
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.multiply(y_conv, 1, name='y_conv')
        tf.summary.histogram('output', y_conv)

    return y_conv


# tf.summary.histogram表示将变量以直方图的形式记录下来，一般用于记录W,b
# tf.summary.scalar表示将数据以标量形式记录下来，一般用于记录loss,accuracy
# 同样两者只是定义数据记录的形式，具体得数据内容需要sess.run来填充
def cnn_train():
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x_input')
        y = tf.placeholder(tf.float32, [None, 10], name='y_input')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    y_conv = cnn(x, keep_prob)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', cross_entropy)

    with tf.name_scope('adam'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                                      tf.argmax(y, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction,name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    # 将tf.summary记录整合
    merged = tf.summary.merge_all()
    # 记录文件得存放路径
    train_writer = tf.summary.FileWriter('./logs')
    # 将当前得默认流图框图保存在记录文件中
    train_writer.add_graph(tf.get_default_graph())

    images, labels = load_mnist('./mnist', kind='train')
    images_test, labels_test = load_mnist('./mnist', kind='t10k')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        # 将一维向量labels转化为tf的one-hot形式，one_hot函数只接受一维向量
        labels = tf.one_hot(labels, 10)
        # 将tensor转化为ndarray形式，因为feed_dict不接受tensor
        labels = labels.eval(session=sess)
        n = 0
        for i in range(2000):
            xs = images[n:n + 100, :]
            ys = labels[n:n + 100, :]
            train_step.run(feed_dict={x: xs, y: ys, keep_prob: 0.5})

            n += 100
            if n >= 60000:
                n -= 60000

            if i % 100 == 0:
                saver.save(sess, './model/cnn_model/cnn.ckpt')
                # keep_prob=0.5表示神经网络中的全连接层中有一半的节点保持激活状态，即能使用，能更新。训练时<1.0
                rs = sess.run(merged, feed_dict={x: xs, y: ys, keep_prob: 0.5})
                # 每100次记录一次数据变化
                train_writer.add_summary(rs, i)
                # keep_prob=1.0表示所有的节点都参与运算，此时dropout=0.0，用于测试
                train_accuracy = sess.run(accuracy,
                                          feed_dict={x: images[0:3000, :], y: labels[0:3000, :], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

        images_test = images_test[0:3000, :]
        labels_test = tf.one_hot(labels_test, 10)
        labels_test = labels_test.eval(session=sess)
        labels_test = labels_test[0:3000, :]
        print('final accuracy: %g' % sess.run(accuracy, feed_dict={x: images_test, y: labels_test, keep_prob: 1.0}))



if __name__=='__main__':
    im = Image.open("/home/liuu/图片/4.png").convert('L')
    im = im.resize((28, 28))
    im = np.array(im)
    im = 255.0 - im
    im = np.reshape(im, [1, 784])

    images_test, labels_test = load_mnist('./mnist', kind='t10k')
    images_test = images_test[0:1000, :]
    labels_test=labels_test[0:1000]

    #result=prediction(images_test)
    result=reload_model(images_test)

    acc=np.equal(result,labels_test)
    acc=acc.astype(np.float32)
    acc=np.mean(acc)

    print(acc)