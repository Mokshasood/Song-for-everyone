import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import cv2
import joblib

path='./CK+48'
img_data=[]
cnt=0
labels=[]
#labels = np.ones((img_data.shape[0],),dtype='int64')
for i in os.listdir(path):
    for j in os.listdir(path+'\\'+i):
        img=cv2.imread(path+'\\'+i+'\\'+j)
        img_resize=cv2.resize(img,(128,128))
        img_data.append(img_resize)
        labels.append(cnt)
    cnt+=1
img_data = np.array(img_data)
#img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 7

names = ['ANGRY','CONTEMPT','DISGUST','FEAR','HAPPY','SAD','SURPRISE']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(img_data,labels,test_size=0.2)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(x_train, y_train, validation_split=0.33, epochs=50, callbacks=callbacks_list, verbose=0)
  
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('model_keras1.h5')
load_model('model_keras.h5')