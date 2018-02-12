import numpy as np
import tensorflow as tf


# 读取mnist文件，并存为ndarray类型的images和labels
# images为60000×784（60000个样本，每个样本784个特征（28×28））
# labels为大小是60000一维向量
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


# 将图片可视化
def show(x_train, y_train):
    import matplotlib.pyplot as plt

    # 在pyplot窗口中同时展示10张图片（2行，每行5张）
    fig, ax = plt.subplots(nrows=2, ncols=5,
                           sharex=True, sharey=True)
    ax = ax.flatten()

    for i in range(10):
        img = x_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    plt.tight_layout()
    plt.show()


# 用dnn算法训练mnist数据
def dnn_mnist(isTrain):
    images, labels = load_mnist('./mnist')

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    w1 = tf.Variable(tf.truncated_normal(shape=[784, 500],
                                         stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[500]))
    y1 = tf.matmul(x, w1) + b1

    w2 = tf.Variable(tf.truncated_normal(shape=[500, 10],
                                         stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[10]))
    y2 = tf.matmul(y1, w2) + b2

    # 将输出结果y2的softmax过程和计算交叉熵过程合并到一起
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y2)
    train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y2, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        if isTrain:
            init = tf.initialize_all_variables()
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=1)
            # 将一维向量labels转化为tf的one-hot形式，one_hot函数只接受一维向量
            labels = tf.one_hot(labels, 10)
            # 将tensor转化为ndarray形式，因为feed_dict不接受tensor
            labels = labels.eval(session=sess)

            n=0
            # 每次只取200个样本进行训练
            for i in range(2000):
                xs = images[n:n + 200, :]
                ys = labels[n:n + 200, :]
                sess.run(train_step, feed_dict={x: xs, y: ys})

                max_acc=0
                if i%100==0:
                    acc = sess.run(accuracy, feed_dict={x: images, y: labels})
                    print('第%i次训练，准确率为：'%i,acc)
                    if i>500 and acc>max_acc:
                        saver.save(sess, './model/ann.ckpt')

                n += 200
                if n >= 60000:
                    n -= 60000

        # 预测
        model = tf.train.latest_checkpoint('model/')
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, model)

        test_images, test_labels = load_mnist('./mnist', 't10k')
        test_labels = tf.one_hot(test_labels, 10)
        test_labels = test_labels.eval(session=sess)

        print('最终训练准确率为：',
              sess.run(accuracy, feed_dict={x: test_images, y: test_labels}))


if __name__ == '__main__':
    dnn_mnist(isTrain=False)
