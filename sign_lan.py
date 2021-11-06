# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:36:48 2021

@author: nsenanay
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

train= pd.read_csv('data/sign_mnist_train.csv')
test= pd.read_csv('data/sign_mnist_test.csv')


train_data= np.array(train, dtype='float32')
test_data=np.array(test, dtype='float32')


class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G' ,'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W','X','Y','Z']

i=random.randint(1, train.shape[0])
fig1,ax1= plt.subplots(figsize=(2,2))
plt.imshow(train_data[i,1:].reshape((28,28)), cmap='gray')
print (class_names[int(train_data[i,0])])



fig= plt.figure(figsize=(18,18))
ax1=fig.add_subplot(221)
train['label'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Lables')


X_train = train_data[:,1:]/255.
X_test= test_data[:,1:]/255.

y_train = train_data[:,0]
y_test= test_data[:,0]


y_train_cat = to_categorical(y_train, num_classes=25)
y_test_cat= to_categorical(y_test, num_classes=25)


X_train=X_train.reshape(X_train.shape[0], *(28,28,1))
X_test= X_test.reshape(X_test.shape[0], *(28,28,1))




model=Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add (Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
          
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(25,activation='softmax'))





model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

history= model.fit(X_train,y_train_cat, batch_size=128, epochs=10, verbose=1, validation_data=(X_test,y_test_cat))

loss=history.history['loss']
val_loss= history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss,'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epoachs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc=history.history['acc']
val_acc= history.history['val_acc']

plt.plot(epochs,acc,'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation_acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

















