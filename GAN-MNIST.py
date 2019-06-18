import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from 传智播客.对抗生成网络.showResult import view_samples



from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/data')


def get_inputs(real_size,noise_size):
    real_img=tf.placeholder(tf.float32,[None,real_size])
    noise_img=tf.placeholder(tf.float32,[None,noise_size])
    return real_img,noise_img


def get_generator(noise_img,n_units,out_dim,reuse=False,alpha=0.01):
    '''
    生成器
    :param noise_img: 产生的噪声输入
    :param n_units: 隐层单元个数
    :param out_dim: （28 28 1）
    :param reuse:
    :param alpha:
    :return:
    '''
    with tf.variable_scope('generator',reuse=reuse):
        # hidden layer  noise_img 100 dim
        hidden1 = tf.layers.dense(noise_img,n_units)
        # leaky ReLU
        hidden1 = tf.maximum(alpha*hidden1,hidden1)
        # dropout
        hidden1 = tf.layers.dropout(hidden1,rate=0.2)

        # logits & outputs 如果输出层是
        logits = tf.layers.dense(hidden1,out_dim)
        output=tf.tanh(logits)

        return logits,output

def get_discriminator(img,n_units,reuse=False,alpha=0.1):
    '''
    判别器
    :param img:
    :param n_units:
    :param reuse: 有两种不同的输入
    :param alpha:
    :return:
    '''
    with tf.variable_scope('discriminator',reuse=reuse):
        # hidden layer
        hidden1=tf.layers.dense(img,n_units)
        hidden1=tf.maximum(alpha*hidden1,hidden1)

        # logits & outputs
        logits = tf.layers.dense(hidden1,1)
        outputs = tf.sigmoid(logits)

        return logits,outputs


img_size = mnist.train.images[0].shape[0]
noise_size = 100
g_units =  128
d_units = 128
learning_rate = 0.001
alpha = 0.01

tf.reset_default_graph()

# 构建网络
real_img , noise_img = get_inputs(img_size,noise_size)
g_logits , g_outputs = get_generator(noise_img, g_units, img_size)
d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)

# region loss
## discriminator的
# 识别真实图片
#  tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
# tf.zeros_like(tensor)  # [[0, 0, 0], [0, 0, 0]]
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                     labels=tf.ones_like(d_logits_real)))
# 识别生成的图片
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                   labels=tf.zeros_like(d_logits_fake)))
# 总体loss
d_loss = tf.add(d_loss_real,d_loss_fake)

## generator的
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)))
# endregion

# region 优化器
train_vars = tf.trainable_variables()
# generator
g_vars = [var for var in train_vars if var.name.startswith('generator')]
# discriminator
d_vars = [var for var in train_vars if var.name.startswith('discriminator')]

# optimizer
d_train_op = tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars)
g_train_op=tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars)
# endregion


batch_size = 64
epoches = 300 # 迭代次数
n_sample = 25 # 抽取样本数
samples = [] # 存储测试样例
losses = [] # 存储loss
saver = tf.train.Saver(var_list=g_vars)
isTrain=False
if(isTrain):
    # region 训练
    # 开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epoches):
            for batch_i in range(mnist.train.num_examples // batch_size):
                batch = mnist.train.next_batch(batch_size)

                batch_images = batch[0].reshape(batch_size, 784)
                # 对图像像素进行scale，这是因为tanh输出结果介于[-1,1]之间，real和fake图片共享discrimination的参数
                batch_images = batch_images * 2 - 1
                # generator的输入噪声
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
                # Rum optimizers
                _ = sess.run(d_train_op, feed_dict={real_img: batch_images, noise_img: batch_noise})
                _ = sess.run(g_train_op, feed_dict={noise_img: batch_noise})

            # 每一轮结束计算loss
            train_loss_d = sess.run(d_loss, feed_dict={real_img: batch_images, noise_img: batch_noise})
            # real img loss
            train_loss_d_real = sess.run(d_loss_real, feed_dict={real_img: batch_images, noise_img: batch_noise})
            # fake img loss
            train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img: batch_images, noise_img: batch_noise})
            # generator loss
            train_loss_g = sess.run(g_loss, feed_dict={noise_img: batch_noise})

            print("Epoch {}/{}...".format(e + 1, epoches),
                  "判别器损失：{:.4f}(判别真实的: {:.4f} + 判别失败的：{:.4f})...".format(train_loss_d, train_loss_d_real,
                                                                         train_loss_d_fake),
                  "生成器损失：{:.4f}".format(train_loss_g)
                  )
            losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

            # 保存样本
            sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
            gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                                   feed_dict={noise_img: sample_noise})

            samples.append(gen_samples)

            saver.save(sess, './checkpoints/generator.ckpt')

    # 保存到本地
    with open('train_samples.pkl', 'wb') as f:
        pickle.dump(samples, f)
    # endregion
else:
    # region 测试
    saver = tf.train.Saver(var_list= g_vars)
    checkpoints = tf.train.get_checkpoint_state('checkpoints')
    if checkpoints and checkpoints.model_checkpoint_path:
        with tf.Session() as sess:
            saver.restore(sess, checkpoints.model_checkpoint_path)
            sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
            gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse= True),
                                   feed_dict={noise_img:sample_noise})
            view_samples(0,[gen_samples])
            plt.show()
    else:
        print("No trained model in: {}".format(os.path.abspath('/checkpoints')))

    # endregion




