import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
from cifar_dataset import get_cifar
from model import *

from scipy.spatial.distance import cdist

cifar = get_cifar()
train_images = np.array([im.reshape((3,32,32)).transpose((1,2,0)) for im in cifar.train.images])
test_images = np.array([im.reshape((3,32,32)).transpose((1,2,0)) for im in cifar.test.images])
len_test = len(cifar.test.images)
len_train = len(cifar.train.images)

img_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3], name='img')
net = mynet(img_placeholder, reuse=False)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state( os.path.join(os.getcwd(), "model_cifar"))
    saver.restore(sess,  os.path.join(os.getcwd(), "model_cifar/model.ckpt"))

    train_feat = sess.run(net, feed_dict={img_placeholder:train_images[:20000]})


true = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state( os.path.join(os.getcwd(), "model_cifar"))
    saver.restore(sess,  os.path.join(os.getcwd(), "model_cifar/model.ckpt"))
    for i in range(len_test):
        im = test_images[i]
        search_feat = sess.run(net, feed_dict={img_placeholder:[im]})
        dist = cdist(train_feat, search_feat, 'cosine')
        rank = np.argsort(dist.ravel())
        if cifar.test.labels[i] == cifar.train.labels[rank[0]]:
            true += 1
    print('Accuracy on test set is: ', true/len_test)
