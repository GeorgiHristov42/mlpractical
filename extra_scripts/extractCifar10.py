import mxnet as mx
import numpy as np
import pickle
import cv2

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f,encoding='bytes')
    print(list(dict.keys()))
    images = dict[b'data']
    images = np.reshape(images, (50000, 3, 32, 32))
    labels = dict[b'fine_labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray, labels

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f,encoding='bytes')
    print(list(dict.keys()))
    return dict[b'fine_label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".jpg", array)

import os
for i in range(1,101):
    path = './cifar/{}'.format(i)
    if not os.path.exists(path):   
       os.mkdir(path)

imgarray, lblarray, labels = extractImagesAndLabels("cifar-100-python/", "train")
print (imgarray.shape)
print (lblarray.shape)

categories = extractCategories("cifar-100-python/", "meta")
numbering = np.zeros(100)

cats = []
for i in range(0,50000):
    print(numbering[labels[i]])
    saveCifarImage(imgarray[i], "./cifar/"+str(labels[i])+"/", "%d"%(numbering[labels[i]]))
    numbering[labels[i]] += 1
    category = lblarray[i].asnumpy()
    category = (int)(category[0])
    cats.append(categories[category])
print (cats)
