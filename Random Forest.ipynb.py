#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[7]:


data=pd.read_csv("hd.csv")


# In[8]:


data.head(5)


# In[11]:


list(data.columns)


# In[20]:


X=data.loc[:,['sqft_living','sqft_lot','sqft_above','sqft_basement']]
X.head(5)
type(X)


# In[21]:


y=pd.DataFrame(data.iloc[:,2])
type(y)


# In[26]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20,criterion="gini",random_state=1,max_depth=3)


# In[30]:


classifier.fit(Xtrain,ytrain)


# In[31]:


y_pred=classifier.predict(Xtest)


# In[32]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[36]:


print(accuracy_score(ytest,y_pred))


# In[ ]:




