#!/usr/bin/env python
# coding: utf-8

# #NUMPY LIBRARY 

# In[6]:


import numpy as np


# In[2]:


x = np.array([[2,3],[6,8]])
y = np.array([[5,6],[2,5]])


# In[3]:


v = np.array([[34,78],[21,65]])
w = np.array([[2,6],[1,4]])


# In[4]:


print(np.dot(v,w),"\n")


# In[5]:


print(np.dot(x,y),'\n')


# # Scikit-learn 

# In[7]:


from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[8]:


dataset = datasets.load_iris()


# In[10]:


model = DecisionTreeClassifier()
model.fit(dataset.data,dataset.target)
print(model)


# In[11]:


expected = dataset.target
predicted = model.predict(dataset.data)


# In[12]:


print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# # Pandas
# 

# In[14]:


import pandas as pd


# In[15]:


data = { "country": ["pakistan","india","china","africa"],
        "capital": ["islamabad","dehli","xxx","yyy"]}
data_table = pd.DataFrame(data)
print(data_table)


# # Matplotlib 

# In[16]:


import matplotlib.pyplot as plt


# In[17]:


import numpy as np


# In[18]:


x = np.linspace(0, 10, 100)


# In[19]:


plt.plot(x, x, label ='linear')


# In[20]:


plt.legend()


# In[22]:


plt.show()


# In[ ]:




