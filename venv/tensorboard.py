#coding = utf-8
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from socketserver import ThreadingMixIn

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist.mnist_deep import *

"""
首先载入tensorflow，并设置训练的最大步数为1000，学习率为0.001，dropout的保留比率为0.9
同时，设置MNISI数据下载地址data_dir和汇总数据的日志存放路径log_dir
这里的日志路径log_dir非常重要，会存放所有汇总数据供TensorFlow展示
"""

from tensorflow.examples.tutorials.mnist import input_data

#死锁
def wait_on_b():
    time.sleep(5)
    #print(b.result())  # b不会完成，他一直在等待a的return结果
    return 5


def wait_on_a():
    time.sleep(5)
    #print(a.result())  # 同理a也不会完成，他也是在等待b的结果
    return 6


# executor = ThreadPoolExecutor(max_workers=2)
# a = executor.submit(wait_on_b)
# b = executor.submit(wait_on_a)

hostIP = ''
portNum = 8080
class mySoapServer(BaseHTTPRequestHandler):
    def do_head(self):
        pass

    def do_GET(self):
        try:
            self.send_response(200, message=None)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            res = '''
            <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">

           <HTML>

           <HEAD><META content="IE=5.0000" http-equiv="X-UA-Compatible">

           <META content="text/html; charset=gb2312" http-equiv=Content-Type>

           </HEAD>

           <BODY>

           Hi, www.perlcn.com is a good site to learn python!

           </BODY>

           </HTML>

           '''
            self.wfile.write(res.encode(encoding='utf_8',errors='strict'))
        except IOError:
            self.send_error(404, message=None)

    def do_POST(self):
        try:
            self.send_response(200, message=None)
            self.send_header('Content-type', 'text/html')
            self.send_header()
            res='''
             <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">

             <HTML>

             <HEAD><META content="IE=5.0000" http-equiv="X-UA-Compatible">

             <META content="text/html; charset=gb2312" http-equiv=Content-Type>

             </HEAD>

             <BODY>

             Hi, www.perlcn.com is a good site to learn python!

             </BODY>

             </HTML>

             '''
            self.wfile.write(res.encode(encoding='utf_8', errors='strict'))
        except IOError:
            self.send_error(404, message=None)

class ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    pass

# myServer = ThreadingHttpServer((hostIP, portNum), mySoapServer)
#
# myServer.serve_forever()
# myServer.server_close()

# a = tf.constant([[1.,2.,3.],[4.,5.,6.]])
# b = np.float32(np.random.randn(3, 2))
# c = tf.matmul(a, b)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# with tf.Session() as sess:
#     print(c.eval())
# sess = tf.InteractiveSession()
# print(c.eval())

max_step = 1000
learning_rate = 0.001
dropout = 0.9
data_dir = './tensorflow/mnist/input_data'
log_dir = 'tensorflow/mnist/logs/mnist_with_summaries'

mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

"""
为了在TensorBoard中展示节点名称，设计网络时会常使用tf.name_scope限制命名空间，
在这个with下所有的节点都会自动命名为input/xxx这样的格式
定义输入x和y的placeholder，并将输入的一维数据变形为28*28的图片存储到另一个tensor，
这样就可以使用tf.summary.image将图片数据汇总给TensorBoard展示
"""

with tf.name_scope('mnist'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

with tf.name_scope('mnist_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

"""
计算出Variable的mean，stddev，max和min
对这些标量数据使用tf.summary.scalar进行记录和汇总
同时，使用tf.summary.histogram直接记录变量var的直方图
"""
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#设计一个MLP多层神经网络来训练数据，在每一层中都会对模型参数进行数据汇总
"""
定义一个传经一层神经网络并进行数据汇总的函数nn_layer
这个函数的输入参数有输入数据input_tensor，输入的维度input_dim，输出的维度output_dim和层名称layer_name,激活函数act则默认使用relu
在函数内，显示初始化这层神经网络的权重和偏置，并使用前面定义的variable_summaries对variable进行数据汇总
然后对输入做矩阵乘法并加上偏置，再将未进行激活的结果使用tf.summary.histogram统计直防图
同时，在使用激活函数后，再使用tf.summary.histogram统计一次
"""

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='actvations')
        tf.summary.histogram('activations', activations)
        return activations

"""
使用刚定义好的nn_layer创建一层神经网络，输入维度是图片的尺寸（784=24*24），输出的维度是隐藏节点数500
再创建一个Droput层，并使用tf.summary.scalar记录keep_prob。然后再使用nn_layer定义神经网络的输出层，
激活函数为全等映射，此层暂时不使用softmax，后面处理
"""

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
y1 = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

"""
这里使用tf.nn.softmax_cross_entropy_with_logits()对前面输出层的结果进行softmax处理并计算交叉熵损失cross_entropy
计算平均损失，并使用tf.summary.saclar进行统计汇总
"""
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

"""
使用Adma优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuray，
再使用tf.summary.scalar对accuracy进行统计汇总
"""
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(y1, 1), tf.arg_max(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

"""
由于之前定义了非常多的tf.summary的汇总操作，意义执行这些操作很麻烦
所以这里使用tf.summary.merge_all()直接获取所有汇总操作，以便后面执行
然后，定义两个tf.summary.FileWrite(文件记录器)在不同的子目录，分别用来存放训练和测试的日志数据
同时，将Session的计算图sess.graph加入训练过程的记录器，这样在TensorBoard的GRAPHS窗口中就能展示整个计算图的可视化效果
最使用tf.global_variables_initializer().run()初始化全部变量
"""

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
tf.global_variables_initializer().run()

"""
定义feed_dict的损失函数
该函数先判断训练标记，如果训练标记为true，则从mnist.train中获取一个batch的样本，并设置dropout值；
如果训练标记为False，则获取测试数据，并设置keep_prob为1，即等于没有dropout效果
"""

def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x:xs, y:ys, keep_prob:k}

#实际执行具体的训练，测试及日志记录的操作
"""
首先，使用tf.train.Saver()创建模型的保存期
然后，进入训练的循环中，每隔10步执行一次merged（数据汇总），accuracy（求测试集上的预测准确率）操作，
并使用tf.RunMetadata()定义Tensorflow运行的元信息
这样可以记录训练是运算时间和内存占用等方面的信息
再执行merged数据汇总操作和train_step训练操作，将汇总summary和训练元信息run_metadata添加到train_writer
平时，则执行merged操作和train_step操作，并添加summary到trian_writer
所有训练全部结束后，关闭train_writer和test_writer
"""

saver = tf.train.Saver()
for i in range(max_step):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict(True),
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d'%i)
            train_writer.add_summary(summary, i)
            saver.save(sess, log_dir+"/model.ckpt", i)
            print('Adding run metadata for', i)
else:
    summary, _=sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()









