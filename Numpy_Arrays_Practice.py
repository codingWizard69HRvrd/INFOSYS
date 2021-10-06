#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This is a comment Gig
#Below are important components of the "numpy" library within Python 3
import numpy as np


# In[2]:


np.ones((3,3,3))


# In[5]:


np.zeros(3)


# In[6]:


np.linspace(12,5,9)


# In[13]:


#linspace creates evenly spaced numbers based on a starting point, end point, and how many numbers it will output
np.linspace(0,100,90)


# In[9]:


np.linspace(0,100,100)


# In[20]:


print('SPY is trading at 430') 


# In[11]:


#np.random.rand will generate an array of random numbers based on a uniform distribution
#This is very, very important in the context of market algorithms
np.random.rand(5)


# In[19]:


#Can be used to create 2+ dimensional arrays
np.random.rand(5,6)


# In[17]:


#identity matrix
np.eye(6)


# In[21]:


#randn is different than rand - randn is based on the STANDARD normal distribution as opposed to the uniform distribution
np.random.randn(5,2)


# In[22]:


#random integers
np.random.randint(1,100)


# In[23]:


np.random.randint(1,20,3)


# In[25]:


#seed- let's say we want to retrieve these "random" values - we use seed as illustrated below
np.random.seed(5)
np.random.rand(3)


# In[26]:


np.random.seed(5)
np.random.rand(3)


# In[30]:


arr = np.arange(25)
ranarr = np.random.randint(0,50,10)


# In[31]:


arr


# In[32]:


arr.shape


# In[33]:


arr.dtype


# In[38]:


arr.reshape(25,1)


# In[ ]:




