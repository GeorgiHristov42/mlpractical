import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

from cifar_dataset import BatchGenerator, get_cifar
from dataset import BatchGenerator, get_mnist

mnist = get_mnist()
cifar = get_cifar()

print('CIFAR: ', cifar.train.images.shape, cifar.train.labels.shape)
print('MNIST: ', mnist.train.images.shape, mnist.train.labels.shape)
