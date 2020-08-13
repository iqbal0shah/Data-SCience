#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


suv=pd.read_csv("suv.csv")


# In[11]:


suv=suv.drop(["User ID","Gender"],axis=1)


# In[12]:


suv.head(5)


# In[13]:


X=suv.iloc[:,:-1].values
y=suv.iloc[:,-1].values


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[18]:


from sklearn.preprocessing import StandardScaler


# In[19]:


sc=StandardScaler()


# In[20]:


X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[23]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)


# In[24]:


classifier.fit(X_train,y_train)


# In[25]:


y_pred=classifier.predict(X_test)


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[29]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[ ]:




