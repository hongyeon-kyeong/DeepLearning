
# coding: utf-8

# In[1]:


#CIFAR-10 dataset
#[URL] https://www.cs.toronto.edu/~kriz/cifar.html


# In[2]:


from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[2]:


import matplotlib.pyplot as plt
from PIL import Image
plt.figure(figsize=(10,10))

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for i in range(0,40) :
    im = Image.fromarray(X_train[i])
    plt.subplot(5,8,i+1)
    plt.title(labels[y_train[i][0]])
    plt.tick_params(labelbottom="off", bottom="off")
    plt.tick_params(labelleft="off", left="off")
    plt.imshow(im)

plt.show()


# In[3]:


X_train.shape


# In[4]:


plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[5]:


len(y_train)


# In[6]:


y_train


# In[7]:


X_train = X_train.reshape(-1, 32*32*3)/255


# In[8]:


import keras
y_train = keras.utils.to_categorical(y_train, 10)


# In[9]:


X_test = X_test.reshape(-1,32*32*3)/255


# In[10]:


y_test = keras.utils.to_categorical(y_test, 10)


# In[11]:


from keras.models import Sequential
from keras.layers import Dense

#입력 데이터 크기 : 32*32 픽셀, RGB형 이미지 데이터 (데이터 전처리 포스팅 참고)
in_size = 32*32*3 

#출력 데이터 크기 : 10개의 카테고리
num_classes = 10

model = Sequential()
#입력층 생성
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
#출력층 생성
model.add(Dense(num_classes,activation='softmax'))


# In[12]:


model.compile(
    #[1]
    loss = 'categorical_crossentropy',
    #[2]
    optimizer = 'adam',
    #[3]
    metrics = ['accuracy']
)


# In[13]:


hist = model.fit(X_train, y_train,
                batch_size=32, #[1]
                epochs=50, #[2]
                verbose=1,
                validation_data=(X_test, y_test)) #[3]


# In[14]:


score = model.evaluate(X_test, y_test, verbose=1)
print('정답률 = ', score[1],'loss=', score[0])


# In[15]:


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy')
plt.legend(['train','test'], loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[16]:


model.save_weights('cifar10-weight.h5')


# In[17]:


import cv2
im = cv2.imread('cat.jpg')


# In[18]:


import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()


# In[19]:


import numpy as np
im = im.reshape(-1,in_size)/ 255
im


# In[20]:


from keras.models import Sequential
from keras.layers import Dense

#입력 데이터 크기 : 32*32 픽셀, RGB형 이미지 데이터 (데이터 전처리 포스팅 참고)
in_size = 32*32*3 

#출력 데이터 크기 : 10개의 카테고리
num_classes = 10

model = Sequential()
#입력층 생성
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
#출력층 생성
model.add(Dense(num_classes,activation='softmax'))

model.load_weights('cifar10-weight.h5')

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# In[21]:


r = model.predict(im, batch_size=32, verbose=1)
r


# In[23]:


res = r[0]
res


# In[24]:


for i, acc in enumerate(res) :
    print(labels[i], "=", int(acc*100))
print("---")
print("예측한 결과 = " , labels[res.argmax()])

