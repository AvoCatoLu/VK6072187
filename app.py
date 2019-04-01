#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[13]:


get_ipython().system(' pip install xlrd')


# In[14]:


print(numpy.__version__,"\n")
print(matplotlib.__version__,"\n")
print(pd.__version__,"\n")
print(tensorflow.__version__,"\n")
print(keras.__version__,"\n")
print(sklearn.__version__,"\n")


# In[15]:


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,1])
    return numpy.array(dataX), numpy.array(dataY)


# In[16]:


df = pd.read_excel("MW.xlsx")


# In[17]:


plt.plot(df)
plt.show()


# In[18]:


df = df.values
df = df.astype('float32')


# In[19]:


print(df)


# In[28]:


train, test = df[0:90,:], df[90:99,:]
print(train,"\n\n", test)


# In[29]:


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(trainX,"\n", trainY,"\n", testX,"\n", testY)


# In[30]:


trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[31]:


print(trainX,"\n\n",testX)


# In[32]:


look_back = 1
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# In[33]:


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[26]:


print(trainPredict,"\n","\n", testPredict)


# In[34]:


a = [92, 93, 94, 95, 96, 97, 98]
result=[(a[0]*testPredict[5]), (a[1]*testPredict[5]), (a[2]*testPredict[5]), (a[3]*testPredict[5]), (a[4]*testPredict[5]), (a[5]*testPredict[5]), (a[6]*testPredict[5]), ]
print(result)


# In[35]:


date = [1080402, 1080403, 1080404, 1080405, 1080406, 1080407, 1080408, ]


# In[39]:


date = pd.Series(date)
result = pd.Series(result)
d = {'date':date, 'MW':result}
submission = pd.DataFrame(data=d)
print(submission)


# In[40]:


submission.to_csv("submission", sep='\t', encoding='utf-8')


# In[ ]:




