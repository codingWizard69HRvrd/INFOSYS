#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright Pierian Data</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # NumPy 
# 
# NumPy is a powerful linear algebra library for Python. What makes it so important is that almost all of the libraries in the <a href='https://pydata.org/'>PyData</a> ecosystem (pandas, scipy, scikit-learn, etc.) rely on NumPy as one of their main building blocks. Plus we will use it to generate data for our analysis examples later on!
# 
# NumPy is also incredibly fast, as it has bindings to C libraries. For more info on why you would want to use arrays instead of lists, check out this great [StackOverflow post](http://stackoverflow.com/questions/993984/why-numpy-instead-of-python-lists).
# 
# We will only learn the basics of NumPy. To get started we need to install it!

# ## *Note*: Numpy Installation Instructions
# 
# ### NumPy is already included in your environment! We HIGHLY recommend using our environment as shown in the setup and installation lecture. You are good to go if you are using the course environment!
# 
# _____
# ##### For those not using the provided environment:
# 
# **It is highly recommended you install Python using the Anaconda distribution to make sure all underlying dependencies (such as Linear Algebra libraries) all sync up with the use of a conda install. If you have Anaconda, install NumPy by going to your terminal or command prompt and typing:**
#     
#     conda install numpy
#     
#     
# **If you do not have Anaconda and can not install it, please refer to [Numpy's official documentation on various installation instructions.](https://www.scipy.org/install.html)**
# 
# _____

# ## Importing NumPy
# 
# Once you've installed NumPy you can import it as a library:

# In[2]:


import numpy as np


# NumPy has many built-in functions and capabilities. We won't cover them all but instead we will focus on some of the most important aspects of NumPy: vectors, arrays, matrices and number generation. Let's start by discussing arrays.
# 
# # NumPy Arrays
# 
# NumPy arrays are the main way we will use NumPy throughout the course. NumPy arrays essentially come in two flavors: vectors and matrices. Vectors are strictly 1-dimensional (1D) arrays and matrices are 2D (but you should note a matrix can still have only one row or one column).
# 
# ## Why use Numpy array? Why not just a list?
# 
# There are lot's of reasons to use a Numpy array instead of a "standard" python list object. Our main reasons are:
# * Memory Efficiency of Numpy Array vs list
# * Easily expands to N-dimensional objects
# * Speed of calculations of numpy array
# * Broadcasting operations and functions with numpy
# * All the data science and machine learning libraries we use are built with Numpy
# 
# ## Simple Example of what numpy array can do

# In[3]:


my_list = [1,2,3]
my_array = np.array([1,2,3])


# In[4]:


type(my_list)


# In[ ]:





# Let's begin our introduction by exploring how to create NumPy arrays.
# 
# ## Creating NumPy Arrays from Objects
# 
# ### From a Python List
# 
# We can create an array by directly converting a list or list of lists:

# In[2]:


my_list = [1,2,3]
my_list


# In[3]:


np.array(my_list)


# In[4]:


my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix


# In[5]:


np.array(my_matrix)


# ## Built-in Methods to create arrays
# 
# There are lots of built-in ways to generate arrays.

# ### arange
# 
# Return evenly spaced values within a given interval. [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.arange.html)]

# In[6]:


np.arange(0,10)


# In[7]:


np.arange(0,11,2)


# ### zeros and ones
# 
# Generate arrays of zeros or ones. [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.zeros.html)]

# In[8]:


np.zeros(3)


# In[9]:


np.zeros((5,5))


# In[10]:


np.ones(3)


# In[11]:


np.ones((3,3))


# ### linspace 
# Return evenly spaced numbers over a specified interval. [[reference](https://www.numpy.org/devdocs/reference/generated/numpy.linspace.html)]

# In[12]:


np.linspace(0,10,3)


# In[13]:


np.linspace(0,5,20)


# <font color=green>Note that `.linspace()` *includes* the stop value. To obtain an array of common fractions, increase the number of items:</font>

# In[14]:


np.linspace(0,5,21)


# ### eye
# 
# Creates an identity matrix [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.eye.html)]

# In[15]:


np.eye(4)


# ## Random 
# Numpy also has lots of ways to create random number arrays:
# 
# ### rand
# Creates an array of the given shape and populates it with random samples from a uniform distribution over ``[0, 1)``. [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.rand.html)]

# In[16]:


np.random.rand(2)


# In[17]:


np.random.rand(5,5)


# ### randn
# 
# Returns a sample (or samples) from the "standard normal" distribution [σ = 1]. Unlike **rand** which is uniform, values closer to zero are more likely to appear. [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randn.html)]

# In[18]:


np.random.randn(2)


# In[19]:


np.random.randn(5,5)


# ### randint
# Returns random integers from `low` (inclusive) to `high` (exclusive).  [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randint.html)]

# In[20]:


np.random.randint(1,100)


# In[21]:


np.random.randint(1,100,10)


# ### seed
# Can be used to set the random state, so that the same "random" results can be reproduced. [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.seed.html)]

# In[22]:


np.random.seed(42)
np.random.rand(4)


# In[23]:


np.random.seed(42)
np.random.rand(4)


# ## Array Attributes and Methods
# 
# Let's discuss some useful attributes and methods for an array:

# In[24]:


arr = np.arange(25)
ranarr = np.random.randint(0,50,10)


# In[25]:


arr


# In[26]:


ranarr


# ## Reshape
# Returns an array containing the same data with a new shape. [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.reshape.html)]

# In[27]:


arr.reshape(5,5)


# ### max, min, argmax, argmin
# 
# These are useful methods for finding max or min values. Or to find their index locations using argmin or argmax

# In[28]:


ranarr


# In[29]:


ranarr.max()


# In[30]:


ranarr.argmax()


# In[31]:


ranarr.min()


# In[32]:


ranarr.argmin()


# ## Shape
# 
# Shape is an attribute that arrays have (not a method):  [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.shape.html)]

# In[33]:


# Vector
arr.shape


# In[34]:


# Notice the two sets of brackets
arr.reshape(1,25)


# In[35]:


arr.reshape(1,25).shape


# In[36]:


arr.reshape(25,1)


# In[37]:


arr.reshape(25,1).shape


# ### dtype
# 
# You can also grab the data type of the object in the array: [[reference](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.dtype.html)]

# In[38]:


arr.dtype


# In[39]:


arr2 = np.array([1.2, 3.4, 5.6])
arr2.dtype


# # Great Job!
