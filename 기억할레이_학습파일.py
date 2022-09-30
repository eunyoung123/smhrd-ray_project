#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime

categories = [
                "36.0","35.5","35.0","34.5","34.0","33.5","33.0","32.5","32.0","31.5","31.0","30.5","30.0",
                "29.5","29.0","28.5","28.0","27.5","27.0","26.5","26.0","25.5","25.0","24.5","24.0",
                "23.5","23.0","22.5","22.0","21.5","21.0","20.5","20.0","19.5","19.0","18.5",
                "18.0","17.5","17.0","16.5","16.0","15.5","15.0","14.5","14.0","13.5","13.0",
                "12.5","12.0","11.5","11.0","10.5","10.0","9.5","9.0","8.5","8.0","7.5","7.0",
                "6.5","6.0","5.5","5.0","4.5","4.0","3.5","3.0","2.5","2.0","1.5","1.0",'0.5'
             ]

num_classes = len(categories)

cnt = 0
image_w = 224
image_h = 224


# In[2]:


group_folder_path = './train_img/' # 훈련용


X = []
Y = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = group_folder_path + categorie + '/'
 
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir+filename)
            cnt = cnt + 1
            print(cnt)
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/256)
            Y.append(label)
            
X_train = np.array(X)
y_train = np.array(Y)


# In[3]:


group_folder_path = './test_img/' # 테스트


X = []
Y = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = group_folder_path + categorie + '/'
 
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir+filename)
            cnt = cnt + 1
            print(cnt)
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/256)
            Y.append(label)
            
X_test = np.array(X)
y_test = np.array(Y)


# In[4]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[5]:


X_train.shape[0] + X_test.shape[0]


# In[1]:


# import os, re, glob
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# import tensorflow as tf

# group_folder_path = './img_canny_d/'
# categories = [
#                 "36.0","35.5","35.0","34.5","34.0","33.5","33.0","32.5","32.0","31.5","31.0","30.5","30.0",
#                 "29.5","29.0","28.5","28.0","27.5","27.0","26.5","26.0","25.5","25.0","24.5","24.0",
#                 "23.5","23.0","22.5","22.0","21.5","21.0","20.5","20.0","19.5","19.0","18.5",
#                 "18.0","17.5","17.0","16.5","16.0","15.5","15.0","14.5","14.0","13.5","13.0",
#                 "12.5","12.0","11.5","11.0","10.5","10.0","9.5","9.0","8.5","8.0","7.5","7.0",
#                 "6.5","6.0","5.5","5.0","4.5","4.0","3.5","3.0","2.5","2.0","1.5","1.0",'0.5'
#              ]

# num_classes = len(categories)

# cnt = 0
# image_w = 224
# image_h = 224
 
# X = []
# Y = []
 
# for idex, categorie in enumerate(categories):
#     label = [0 for i in range(num_classes)]
#     label[idex] = 1
#     image_dir = group_folder_path + categorie + '/'
 
#     for top, dir, f in os.walk(image_dir):
#         for filename in f:
#             print(image_dir+filename)
#             cnt = cnt + 1
#             print(cnt)
#             img = cv2.imread(image_dir+filename)
#             img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
#             X.append(img/256)
#             Y.append(label)
            
# X = np.array(X)
# Y = np.array(Y)

# X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=3)
# xy = (X_train, X_test, y_train, y_test)

# np.save("../Desktop/raydata/img_ray", xy)


# In[6]:


X_train.shape


# In[7]:


from keras.models import load_model
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping

#X_train, X_test, y_train, y_test = np.load('../Desktop/raydata/img_ray.npy', allow_pickle=True) 

# 모델 구성도
model = Sequential() 
model.add(Conv2D(96, (3,3), input_shape=(224, 224, 3), strides=4 ,activation='relu', padding='same'))        
model.add(MaxPooling2D((2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(384, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(384,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(256,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2),padding='same'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(72, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=200, epochs=100)

model.save('../Desktop/raydata/ray_canny_123.h5')


# In[8]:


X_train = np.append(X_train,X_test, axis=0)
y_train = np.append(y_train,y_test, axis=0)

# Save Model with CheckPoint & StopPoint


Datetime = datetime.datetime.now().strftime('%m%d_%H%M')
modelpath="../Desktop/raydata/ray_canny_123.h5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='loss', patience=50)

# Learning and save models
model.fit(X_train, y_train, validation_split=0.1, epochs=3500, batch_size=15, verbose=0, callbacks=[early_stopping_callback,checkpointer])


# In[10]:


model.evaluate(X_test,y_test)


# In[35]:


import os, re, glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model

categories = [
                 "36.0","35.5","35.0","34.5","34.0","33.5","33.0","32.5","32.0","31.5","31.0","30.5","30.0",
                "29.5","29.0","28.5","28.0","27.5","27.0","26.5","26.0","25.5","25.0","24.5","24.0",
                "23.5","23.0","22.5","22.0","21.5","21.0","20.5","20.0","19.5","19.0","18.5",
                "18.0","17.5","17.0","16.5","16.0","15.5","15.0","14.5","14.0","13.5","13.0",
                "12.5","12.0","11.5","11.0","10.5","10.0","9.5","9.0","8.5","8.0","7.5","7.0",
                "6.5","6.0","5.5","5.0","4.5","4.0","3.5","3.0","2.5","2.0","1.5","1.0"
             ]


def Dataization(img_path):
    image_w = 224
    image_h = 224
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)
    


# In[36]:


src = []
name = []
test = []

image_dir = '../Desktop/test_img_ray/canny/'
for file in os.listdir(image_dir):     
    src.append(image_dir + file)
    name.append(file)
    test.append(Dataization(image_dir + file))
    


# In[37]:


test = np.array(test)
model = load_model('../Desktop/raydata/ray_canny_123.h5')

y_prob = model.predict(test, verbose=0) 
predict = y_prob.argmax(axis=-1)


for i in range(len(test)):
    print(name[i] + " : , Predict : "+ str(categories[predict[i]]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




