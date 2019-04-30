
# coding: utf-8

# In[3]:


### Mutiple Linear Regression
## Boston housing


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


# In[7]:


# Load data
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']


# In[9]:


# Make and fit linear regression model
model = LinearRegression()
model.fit(x, y)


# In[11]:


# Make prediction using the model 
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]

# predict the housing price for sample_house
model.predict(sample_house)

