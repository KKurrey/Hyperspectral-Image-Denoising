import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import spectral as sp
import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import cv2

indiana_pines = np.load('result\indianpinearray.npy')
ground_truth = np.load('result\IPgt.npy')


orig_img = np.asarray(indiana_pines)


orig_img = orig_img.astype('float32')
orig_img /= 9604.0

data =orig_img.reshape((145*145,200))
X_train=data[:,0:160]
X_test=data[:,160:]

X_train=X_train.T
X_test=X_test.T

orig_img_noisy = orig_img + np.random.normal(loc = 0.5, scale = 0.002, size = orig_img.shape)

# Clipping the magnitudes to live within 0 and 1
orig_img_noisy = np.clip(orig_img_noisy, 0, 1)

data_noise=orig_img_noisy.reshape((145*145,200))

# Plotting the image
# view = sp.imshow(orig_img_noisy, (145, 145, 199))

#Model building
input_img = keras.Input(shape=(21025,))
encoded = layers.Dense(256, activation='relu')(input_img)
encoded = layers.Dense(128, activation='relu')(encoded)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(256, activation='relu')(decoded)
decoded = layers.Dense(21025, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#training
autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=20,
                shuffle=True,
                validation_data=(X_test, X_test))

#Testing - input data
decoded_imgs = autoencoder.predict(X_test)
n = 2
plt.figure(figsize=(4, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i].reshape(145, 145))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(145, 145))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#PSNR - input output image
for i in range(1,3):
    p=cv2.PSNR(X_test[i].reshape(145, 145),decoded_imgs[i].reshape(145, 145))
    print(i)
    print(f"PSNR = {p}")
    

# Testing - noisy data
noisy_data=data_noise.T
filt_imgs = autoencoder.predict(noisy_data)
filt_imgs=np.float64(filt_imgs)
n = 2
plt.figure(figsize=(4, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(noisy_data[195+i].reshape(145, 145))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(filt_imgs[112+i].reshape(145, 145))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#PSNR - noisy
for i in range(1,3):
    p=cv2.PSNR(noisy_data[195+i].reshape(145, 145),filt_imgs[112+i].reshape(145, 145))
    print(i)
    print(f"PSNR = {p}")
     
















