#!/usr/bin/env python
# coding: utf-8

# # importing the necessary modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# # Data preprocessing

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Loan status prediction/data.csv')


# In[3]:


#to see the dataset
df.head()


# In[4]:


#shape of the dataset
df.shape


# In[5]:


#to get some insights about the dataset
df.info()


# In[6]:


#to get the statistical measure of the dataset
df.describe()


# In[7]:


#to check the sum of null values in each column
df.isnull().sum()


# In[8]:


#As the null values are less compared to data, we are dropping the null values
df = df.dropna()


# In[9]:


df.isnull().sum()


# In[10]:


df.shape


# Label Encoding

# In[11]:


#convert categorical columns to numerical values
encoder = LabelEncoder()

objList = df.select_dtypes(include = "object").columns
print (objList)
objList = objList.drop('Loan_ID')

for string in objList:
    df[string] = encoder.fit_transform(df[string].astype(str))


# In[12]:


df.head()


# In[13]:


df.shape


# In[14]:


df['Dependents'].value_counts()


# In[15]:


#Replacing the 3+ with 4
df = df.replace(to_replace = '3+', value = 4)


# In[16]:


df['Dependents'].value_counts()


# # Data visualization

# In[17]:


#education and loan status
sns.countplot(x = 'Education', hue = 'Loan_Status', data = df)


# In[18]:


#martial status and loan status
sns.countplot(x = 'Married', hue = 'Loan_Status', data = df)


# In[19]:


#gender and loan status
sns.countplot(x = 'Gender', hue = 'Loan_Status', data= df)


# Separating data and labels

# In[27]:


X = df.drop(columns = ['Loan_ID', 'Loan_Status'], axis =1)
Y = df['Loan_Status']


# In[29]:


X.head()
Y.head()


# Splitting the data into training data and testing data

# In[30]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, stratify = Y, random_state = 2)


# Train the model :
# 
# Support Vector Machine Model

# In[32]:


classifier = svm.SVC(kernel = 'linear')


# In[33]:


#training the support vector model
classifier.fit(x_train, y_train)


# In[34]:


#model evauation
#accuracy score on training data
prediction = classifier.predict(x_train)

accu_score = accuracy_score(prediction, y_train)
print('THE ACCURACY OF THE MODEL IS :', accu_score)


# In[37]:


#model evauation
#accuracy score on testing data
test_prediction = classifier.predict(x_test)

test_accu_score = accuracy_score(test_prediction, y_test)
print('THE ACCURACY OF THE MODEL IS :', test_accu_score)


# # predictive model system

# In[39]:


'''
for this wrds type the corresponding int value

Female,No,Non-Graduate,Rural = 0
Male,Yes,Graduate,Semi-Urban = 1
Urban                        = 2
3+                           = 4

'''

input_data = input()
input_list = [float(i) for i in input_data.split(',')]
input_array = np.asarray(input_list)
reshaped_array = input_array.reshape(1, -1)

model_pred = classifier.predict(reshaped_array)
print('THE PREDICTED VALUE IS :' , model_pred)
if model_pred == 0:
    print('LOAN WILL NOT BE APPROVED')
else:
    print('LOAN WILL BE APPROVED')


# In[ ]:




