import tensorflow as tf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import random
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import choice, permutation
from itertools import combinations

flags = tf.app.flags
FLAGS = flags.FLAGS


class BatchGenerator():
	def __init__(self, images, labels):
		np.random.seed(0)
		random.seed(0)
		self.labels = labels
		print(images.shape)
		im = images.reshape((42500, 3, 32, 32))
		self.images = np.transpose(im, (0, 2, 3, 1))
		self.tot = len(labels)
		self.i = 5
		self.num_idx = dict()
		for idx, num in enumerate(self.labels):
			if num in self.num_idx:
				self.num_idx[num].append(idx)
			else:
				self.num_idx[num] = [idx]
		print(self.num_idx.keys())
		print(self.images.shape)
		self.to_img = lambda x: self.images[x]

	def next_batch(self, batch_size):
		left = []
		right = []
		sim = []
		# genuine
		for i in range(100):
			n = 2
			l = choice(self.num_idx[i], n*2, replace=False).tolist()
			left.append(self.to_img(l.pop()))
			right.append(self.to_img(l.pop()))
			sim.append([1])

		#impostor
		incl = 0
		for i,j in combinations(range(100), 2):
			if incl == 19:
				left.append(self.to_img(choice(self.num_idx[i])))
				right.append(self.to_img(choice(self.num_idx[j])))
				sim.append([0])
				incl = 0
			else:
				 incl += 1
		return np.array(left), np.array(right), np.array(sim)


class Object(object):
    pass


def get_cifar():
	print(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CIFAR100/cifar100-train.npz'))
	data_path = os.path.join(
		os.environ['MLP_DATA_DIR'], 'cifar100-train.npz')
	#data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CIFAR100/cifar100-train.npz')
	train = Object()
	train.images = np.load(data_path)['inputs']
	train.labels = np.load(data_path)['targets']
	data_path = os.path.join(
		os.environ['MLP_DATA_DIR'], 'cifar100-test.npz')
	#data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CIFAR100/cifar100-test.npz')
	test = Object()
	test.images = np.load(data_path)['inputs']
	test.labels = np.load(data_path)['targets']
	cifar = Object()
	cifar.train = train
	cifar.test  = test

	return cifar
