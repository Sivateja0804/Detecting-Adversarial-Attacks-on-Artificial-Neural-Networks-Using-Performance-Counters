#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

from tensorflow.keras.datasets import mnist, cifar10, cifar100

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from tensorflow.keras.optimizers import SGD
import numpy as np
from numpy import save
from numpy import asarray
from numpy import load
import random

import matplotlib.pyplot as plt


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


# In[3]:


img_rows, img_cols, channels = 28, 28, 1
num_classes = 10

x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols, channels))
x_test = x_test.reshape((x_test.shape[0], img_rows, img_cols, channels))

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

print("Data shapes", x_test.shape, y_test.shape, x_train.shape, y_train.shape)


# In[4]:


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

model = create_model()


# In[5]:


model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test))


# In[6]:


m1=tf.keras.models.clone_model(model)
m1.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
m1.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test))
m1.save('final_model_clean.h5')


# In[7]:


loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Base accuracy on regular images:", acc)
print("Base loss on regular images:", loss)


# In[8]:


def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    
    gradient = tape.gradient(loss, image)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad


# In[9]:


'''image = x_train[0]
image_label = y_train[0]
perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label).numpy()
adversarial = image + perturbations * 0.1
if channels == 1:
    plt.imshow(adversarial.reshape((img_rows, img_cols)))
else:
    plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))
plt.show()'''


# In[10]:


'''print(labels[model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()])
print(labels[model.predict(adversarial).argmax()])'''


# In[11]:



def generate_adversarials(batch_size):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
            #N = random.randint(0, 100)

            label = y_test[batch]
            image = x_test[batch]
            
            perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label).numpy()
            
            
            epsilon = 0.1
            adversarial = image + perturbations * epsilon
            
            x.append(adversarial)
            y.append(y_test[batch])
        
        
        x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
        y = np.asarray(y)
        
        yield x, y


# In[12]:


x_adversarial_test, y_adversarial_test = next(generate_adversarials(10000))


# In[13]:


loss1, acc1 = model.evaluate(x_test, y_test, verbose=0)
print("Base accuracy on regular images:", acc1)
print("Base loss on regular images:", loss1)
loss2, acc2 = model.evaluate(x_adversarial_test, y_adversarial_test, verbose=0)
print("Base accuracy on adversarial images:",acc2)
print('Base loss on adversarial images:', loss2)


# In[14]:


save('adversarial_test_images',x_adversarial_test)
save('adversarial_test_labels',y_adversarial_test)
save('clean_test_images',x_test)
save('clean_test_labels',y_test)
model.save('final_model_adv.h5')

