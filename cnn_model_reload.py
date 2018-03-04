import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.python.framework import graph_util

#包含两种模型加载方法

#第一种，最简单的
def reload_model(image_array):

    assert isinstance(image_array,np.ndarray)
    assert image_array.shape[1]==784

    saver=tf.train.import_meta_graph('./model/cnn_model/cnn.ckpt.meta')
    graph=tf.get_default_graph()

    x=graph.get_tensor_by_name('input/x_input:0')
    y=graph.get_tensor_by_name('input/y_input:0')
    y_conv=graph.get_tensor_by_name('output/y_conv:0')
    keep_prob=graph.get_tensor_by_name('input/keep_prob:0')
    accuracy=graph.get_tensor_by_name('accuracy/accuracy:0')


    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint('./model/cnn_model'))
        result = sess.run(y_conv, feed_dict={x: image_array, keep_prob: 1.0})

        #同时输入多个图片时，返回n*10的列表，需要分别找出每一行的最大值位置
        result = np.argmax(result,axis=1)
        return(result)



#第二种，比较复杂的，包含3个函数，但可移植性好

#函数1：将变量转换为常量，与结构图绑定，存为pb文件
def freeze_graph(model_folder):
    # 检查文件夹中有无检查点文件，如果有，返回该文件
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    # 获取检查点文件中名为‘model_checkpoint_path’的str（模型文件名称）
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_folder = '/'.join(input_checkpoint.split('/')[:-1])
    # 在存放checkpoint的文件夹下存放pb文件，.pb文件存放meta + data
    output_graph = absolute_model_folder + '/frozen_model.pb'

    output_node_names = ["output/y_conv", "input/keep_prob","accuracy/accuracy"]
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    # 返回一个图的序列化的GraphDef表示，以便导入另一个图
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # 将output_node_names表示得节点变量及与之相关得节点变量（Variable）用常量（constant）代替
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names
        )

        with tf.gfile.GFile(output_graph, 'wb') as f:
            # 序列化，就是转换格式，转换成流的形式
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


#函数2：读取pb文件，并返回一个图
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        # 先建立一张空图
        graph_def = tf.GraphDef()
        # 从字符串反序列化，读取之前存好的图
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')

    return graph


def prediction(image_array):

    assert isinstance(image_array, np.ndarray)
    assert image_array.shape[1] == 784

    #freeze_graph('./model/cnn_model')
    graph = load_graph('./model/cnn_model/frozen_model.pb')

    with graph.as_default():

        for op in graph.get_operations():
            print(op.name, op.values())

        #prefix/input/x_input后面的:0不能省，因为x_input代表一个操作，x_input:0(<op_name>:<output_index>)才代表一个tensor
        x = graph.get_tensor_by_name('prefix/input/x_input:0')
        #y = graph.get_tensor_by_name('prefix/input/y_input:0')
        y_conv = graph.get_tensor_by_name('prefix/output/y_conv:0')
        keep_prob = graph.get_tensor_by_name('prefix/input/keep_prob:0')
        #accuracy=graph.get_tensor_by_name('prefix/accuracy/accuracy:0')

        with tf.Session() as sess:
            result = sess.run(y_conv, feed_dict={x: image_array, keep_prob: 1.0})
        result = np.argmax(result,axis=1)
        return(result)