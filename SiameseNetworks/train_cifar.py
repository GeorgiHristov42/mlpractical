import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os

from cifar_dataset import BatchGenerator, get_cifar
from model import *

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('train_iter', 20000, 'Total training iter')
flags.DEFINE_integer('step', 10, 'Save after ... iteration')

cifar = get_cifar()
gen = BatchGenerator(cifar.train.images, cifar.train.labels)
test_im = np.array([im.reshape((3,32,32)).transpose((1, 2, 0)) for im in cifar.test.images])
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']


left = tf.placeholder(tf.float32, [None, 32, 32, 3], name='left')
right = tf.placeholder(tf.float32, [None, 32, 32, 3], name='right')
with tf.name_scope("similarity"):
	label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	label = tf.to_float(label)
margin = 1

left_output = mynet(left, reuse=False)

right_output = mynet(right, reuse=True)

loss = contrastive_loss(left_output, right_output, label, margin)

global_step = tf.Variable(0, trainable=False)


# starter_learning_rate = 0.001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.scalar_summary('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

train_step = tf.train.MomentumOptimizer(0.00001, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)


saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#setup tensorboard
	tf.summary.scalar('step', global_step)
	tf.summary.scalar('loss', loss)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('train.log', sess.graph)

	#train iter
	for i in range(FLAGS.train_iter):
		b_l, b_r, b_sim = gen.next_batch(FLAGS.batch_size)

		_, l, summary_str = sess.run([train_step, loss, merged],
			feed_dict={left:b_l, right:b_r, label: b_sim})

		writer.add_summary(summary_str, i)
		print("\r#%d - Loss"%i, l)


		if (i + 1) % FLAGS.step == 0:
			#generate test
			#feat = sess.run(left_output, feed_dict={left:test_im})
			#
			# labels = cifar.test.labels
			# plot result
			# f = plt.figure(figsize=(16,9))
			# for j in range(10):
			#     plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(),
			#     	'.', c=c[j],alpha=0.8)
			# plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
			# plt.savefig('img/%d.jpg' % (i + 1))
			saver.save(sess, os.path.join(os.path.dirname(os.path.realpath(__file__)),"model_cifar/model.ckpt"))
			print('Checkpoin saved')
	saver.save(sess, os.path.join(os.path.dirname(os.path.realpath(__file__)),"model_cifar/model.ckpt"))
