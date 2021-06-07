#!/usr/bin/env python
# coding: utf-8

# In[74]:


from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[75]:





# In[95]:


print('It is a program that predicts monthly income by entering the number of household members, the age of household owners, and the income quintile!')
df = pd.read_csv("data.csv")
df.dropna()

df.head()


# In[77]:


from sklearn.model_selection import train_test_split

x = df[['가구원수','가구주연령', '소득분위']]
y = df[['소득']]


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)


# In[78]:


lr = LinearRegression()
lr.fit(x_train, y_train)


# In[80]:


y_predict = lr.predict(x_test)


# In[93]:


import matplotlib.pyplot as plt

# plt.scatter(y_year, y_rate, alpha=0.4)
plt.scatter(y_test, y_predict, alpha=0.2,
            cmap='viridis')

plt.xlabel("Actual Income")
plt.ylabel("Predict Income")
plt.title("Monthly Income")
# plt.colorbar()
print('The closer the graph is to a straight line, the more accurate it is.')
plt.show()
# plt.show()


# # lr.coef_

# In[94]:


print('Program Accuracy (Closer to 1, More Accurate)')
print(lr.score(x_train, y_train))
print(lr.score(x_test, y_test))


# In[96]:


input1 = input("Please enter the number of household members: ")
input1 = int(input1)

input2 = input("Please enter the age of the householder: ")
input2 = int(input2)

input3 = input("Please enter your income quintile: ")
input3 = int(input3)

result = []
result.append(input1)
result.append(input2)
result.append(input3)
temp = []
temp.append(result)
my_predict = lr.predict(temp)
my_predict = int(my_predict)
print(my_predict, 'won')


# In[ ]:




