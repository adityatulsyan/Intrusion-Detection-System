#!/usr/bin/env python
# coding: utf-8

# # Building Intrusion Detection System using Artificial Neural Networks
# ## Data Clean up and Pre-Processing

# In[1]:


#Importing desired modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv("ids.csv")
df = pd.DataFrame(dataset)
df.info()


# In[3]:


df.columns


# In[4]:


#Checking the shape of the complete datset
df.shape
#Rounding the data to two decimal places
df = df.round(2)


# ### 692703 rows × 79 columns is the size of the original data

# In[5]:


#Replacing infinity values with NaN
df.replace([np.inf, -np.inf], np.nan)
#Removing rows containing NaN 
df.dropna(how="any", inplace = True)


# In[6]:


#Shape after removinf NaNs
df.shape


# In[7]:


# Since the data contains 79 parameters,which will require quite a lot of processing,
# so we are dropping columns with either constant value or very much divergent values
df = df.drop(df.std()[df.std() < .3].index.values, axis=1)
df = df.drop(df.std()[df.std() > 1000].index.values, axis=1)


# In[8]:


#new shape of the dataset after dropping the columns with divergent values
df.shape


# In[9]:


#Various types of labels associated with the dataset
df[' Label'].value_counts()


# <!-- ### Labels associated with data
# BENIGN              439972
# DoS Hulk            230124
# DoS GoldenEye        10293
# DoS slowloris         5796
# DoS Slowhttptest      5499
# Heartbleed              11 -->

# In[10]:


#Pie chart representing share of different type of Label
labels = 'BENIGN', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed'
fig, ax = plt.subplots()
values = df[' Label'].value_counts()
explodeTuple = (0.2, 0.4, 0.6, 0.8, 1.0, 1.4)
ax.pie(values, explode = explodeTuple, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)


# In[11]:


# Since distribution opf various Denial of Service (DoS) is highly irregular, we have clubbed them for better results
df = df.replace('Heartbleed', 'DoS')
df = df.replace('DoS GoldenEye', 'DoS')
df = df.replace('DoS Slowhttptest', 'DoS')
df = df.replace('DoS slowloris', 'DoS')
df = df.replace('DoS Hulk', 'DoS')
df[' Label'].value_counts()


# In[12]:


df = df[~df['Flow Bytes/s'].isin(['Infinity'])]
df = df[~df[' Flow Packets/s'].isin(['Infinity'])]
df.shape


# In[13]:


#Processing aroung 700k data is quite a time consuming task, we are taking only 100k for our training and testing our model
df = df.iloc[:100000]


# In[14]:


#Final shape of the data after cleanup and pre-processing
df.shape


# ### 100000 rows × 24 columns is the size of the new data

# In[15]:


df[' Label'].value_counts()


# In[16]:


#Plotting pie chart of labels associated with row to show ratio of both types of Labels
label = 'BENIGN', 'DoS'
value = df[' Label'].value_counts()
fig1, ax1 = plt.subplots()
explodeTuple = (0.1, 0.1)
ax1.pie(value, explode = explodeTuple, labels = label, autopct='%1.1f%%',
        shadow=True, startangle=90)


# In[17]:


#Importing various packages and libraries 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from IPython.display import SVG
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential
from keras.layers import Dense, Activation


# In[18]:


#Replacing labels with 1 and 0 for convenience
df.replace(to_replace ="BENIGN", value = 1, inplace = True)
df.replace(to_replace ="DoS", value = 0, inplace = True)


# In[19]:


#First five rows of the data
df.head()


# In[20]:


x = df.drop(' Label', 1)
y = df[' Label']


# In[21]:


#Splitting data into train and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[22]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[23]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)


# In[24]:


print('Train images shape:', X_train.shape)
print('Train labels shape:', y_train.shape)
print('Test images shape:', X_test.shape)
print('Test labels shape:', y_test.shape)
print('Train labels:', y_train)
print('Test labels:', y_test)


# In[25]:


model = Sequential()
model.add(Dense(256, activation='relu', input_dim = 23))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))


# In[26]:


# For a binary classification problem
from keras.optimizers import SGD
opt = SGD(lr=0.01)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ['accuracy'])


# In[27]:


history = model.fit(X_train, y_train, epochs=20,
          verbose=1, batch_size=100)


# In[28]:


score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[29]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[30]:


def plot_keras_model(model, show_shapes=True, show_layer_names=True):
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    return SVG(model_to_dot(model, show_shapes=show_shapes, show_layer_names=show_layer_names).create(prog='dot', format='svg'))


# In[31]:


from keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[32]:


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)


# In[33]:


# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

