#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


emp_data = pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')
emp_data.head()


# In[4]:


emp_data.rename(columns={'sales':'dept'}, inplace=True)
emp_data.head()


# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


emp_data.describe()


# # Preprocessing

# In[7]:


emp_data.select_dtypes('object').columns


# In[8]:


emp_data.dept.value_counts()


# In[9]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[10]:


le = LabelEncoder()


# In[11]:


dept = le.fit_transform(emp_data.dept)


# In[12]:


ohe = OneHotEncoder()


# In[13]:


ohe_dept = ohe.fit_transform(dept.reshape(-1,1))


# In[16]:


le.classes_


# In[17]:


dept_df = pd.DataFrame(ohe_dept.toarray(), dtype=int,columns=le.classes_)


# In[18]:


emp_data['salary_tf'] = emp_data.salary.map({'low':1,'medium':2,'high':3})


# In[19]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler


# In[20]:


emp_data.columns


# In[21]:


df = emp_data[['number_project','average_montly_hours', 'time_spend_company']]


# In[22]:


df.plot.kde()


# In[23]:


mm = MinMaxScaler()


# In[24]:


scaled_np = mm.fit_transform(df)


# In[25]:


dept_np = dept_df.values


# In[26]:


emp_df = emp_data[['satisfaction_level','last_evaluation','Work_accident','promotion_last_5years','salary_tf']]


# In[27]:


emp_np = emp_df.values


# In[28]:


feature_data = np.hstack([emp_np, scaled_np, dept_np])


# In[29]:


target_data = emp_data.left


# In[30]:


feature_data.shape


# In[31]:


target_data.value_counts()


# # Model Building

# In[32]:


from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier


# In[33]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


models = [ LogisticRegression(class_weight='balanced'), SGDClassifier(max_iter=10), PassiveAggressiveClassifier(max_iter=20), RandomForestClassifier(n_estimators=20)]


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


trainX,testX,trainY,testY = train_test_split(feature_data,target_data)


# In[37]:


for model in models:
    model.fit(trainX,trainY)
    print (model.score(testX,testY))

