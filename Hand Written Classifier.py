 import numpy as np
 import os
 from tensorflow.keras.datasets import mnist
 from matplotlib import pyplot as plt
 (trainX, trainy), (testX, testy) = mnist.load_data()
 print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
 print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
 Train: X=(60000, 28, 28), y=(60000,)
 Test: X=(10000, 28, 28), y=(10000,)
 for i in range(9):
 plt.subplot(330 + 1 + i)
 plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
 data = ...

 import matplotlib.pyplot as plt
 import seaborn as sns
 #import cv2
 from PIL import Image
 import tensorflow as tf
 tf.random.set_seed(3)
 from tensorflow import keras
 from keras.datasets import mnist
 from tensorflow.math import confusion_matrix
 import random
 (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
 type(X_train)
 numpy.ndarray
 print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
 (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
 print(X_train[9])

 print(X_train[9].shape)
 (28, 28)
 3
 plt.imshow(X_train[50])
 plt.show
 print(Y_train[50])

 print(Y_train.shape,Y_test.shape)
 (60000,) (10000,)


 # print unique value in Y_train
 print(np.unique(Y_train))
 # print unique value in Y_test
 print(np.unique(Y_test))
 [0 1 2 3 4 5 6 7 8 9]
 [0 1 2 3 4 5 6 7 8 9]
 X_train = X_train/255
 X_test = X_test/255
 print(X_train[9])