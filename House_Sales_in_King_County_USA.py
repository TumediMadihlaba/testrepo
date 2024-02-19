#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="300" alt="Skills Network Logo">
#     </a>
# </p>
# 
# <h1 align="center"><font size="5">Final Project: House Sales in King County, USA </font></h1>
# 

# # About the Dataset
# 
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

# | Variable      | Description                                                                                                 |
# | ------------- | ----------------------------------------------------------------------------------------------------------- |
# | id            | A notation for a house                                                                                      |
# | date          | Date house was sold                                                                                         |
# | price         | Price is prediction target                                                                                  |
# | bedrooms      | Number of bedrooms                                                                                          |
# | bathrooms     | Number of bathrooms                                                                                         |
# | sqft_living   | Square footage of the home                                                                                  |
# | sqft_lot      | Square footage of the lot                                                                                   |
# | floors        | Total floors (levels) in house                                                                              |
# | waterfront    | House which has a view to a waterfront                                                                      |
# | view          | Has been viewed                                                                                             |
# | condition     | How good the condition is overall                                                                           |
# | grade         | overall grade given to the housing unit, based on King County grading system                                |
# | sqft_above    | Square footage of house apart from basement                                                                 |
# | sqft_basement | Square footage of the basement                                                                              |
# | yr_built      | Built Year                                                                                                  |
# | yr_renovated  | Year when house was renovated                                                                               |
# | zipcode       | Zip code                                                                                                    |
# | lat           | Latitude coordinate                                                                                         |
# | long          | Longitude coordinate                                                                                        |
# | sqft_living15 | Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area |
# | sqft_lot15    | LotSize area in 2015(implies-- some renovations)                                                            |
# 

# ## **Import the required libraries**

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # Module 1: Importing Data Sets

# Simply using the URL directly in the pandas.read_csv() function. 

# In[27]:


filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(filepath)


# We use the method head to display the first 5 columns of the dataframe.

# In[28]:


df.head()


# ### Question 1
# 
# Display the data types of each column using the function dtypes. Take a screenshot of your code and output.

# In[29]:


# list the data types for each column

print(df.dtypes)


# We use the method describe to obtain a statistical summary of the dataframe.

# In[30]:


df.describe()


# # Module 2: Data Wrangling

# ### Question 2
# 
# Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(), then use the method describe() to obtain a statistical summary of the data. Make sure the inplace parameter is set to True. 

# In[31]:


df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)


# Also, we use the method describe to obtain a statistical summary of the dataframe.

# In[32]:


df.describe()


# We can see we have missing values for the columns  bedrooms and  bathrooms. 
# 

# In[33]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# We can replace the missing values of the column 'bedrooms' with the mean of the column 'bedrooms'  using the method replace(). Don't forget to set the inplace parameter to True

# In[34]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# We also replace the missing values of the column 'bathrooms' with the mean of the column 'bathrooms'  using the method replace(). Don't forget to set the  inplace  parameter top  True 

# In[36]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[37]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# # Module 3: Exploratory Data Analysis

# ### Question 3
# 
# Use the method <code>value_counts</code> to count the number of houses with unique floor values, use the method <code>.to_frame()</code> to convert it to a data frame.

# In[38]:


df['floors'].value_counts()


# In[39]:


df['floors'].value_counts().to_frame()


# ### Question 4
# 
# Use the function <code>boxplot</code> in the seaborn library  to  determine whether houses with a waterfront view or without a waterfront view have more price outliers.

# In[40]:


sns.boxplot(x="waterfront", y="price", data=df)


# Houses without a waterfront view have more price outliers than the houses with a waterfron view.

# ### Question 5
# 
# Use the function <code>regplot</code>  in the seaborn library  to  determine if the feature <code>sqft_above</code> is negatively or positively correlated with price.

# In[41]:


sns.regplot(x="sqft_above", y="price", data=df, line_kws={"color": "red"})
plt.ylim(0,)


# The feature sqft_above is positively correlated with price.

# We can use the Pandas method corr() to find the feature other than price that is most correlated with price.

# In[ ]:


df.corr()['price'].sort_values()


# # Module 4: Model Development

# We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.

# In[45]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# ### Question  6
# 
# Fit a linear regression model to predict the <code>'price'</code> using the feature <code>'sqft_living'</code> then calculate the R^2. 

# In[47]:


X1 = df[['sqft_living']]
Y1 = df['price']
lm1 = LinearRegression()
lm1.fit(X1,Y)
lm1.score(X1, Y)


# In[49]:


Yhat_1=lm1.predict(X1)
print(Yhat_1)


# ### Question 7
# 
# Fit a linear regression model to predict the <code>'price'</code> using the list of features:
# 

# In[50]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     


# Then calculate the R^2. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.

# In[51]:


Z = df[features]
lm = LinearRegression()
lm.fit(Z, Y)
lm.score(Z, Y)


# In[52]:


Yhat=lm.predict(Z)
print(Yhat)


# ### This will help with Question 8
# 
# Create a list of tuples, the first element in the tuple contains the name of the estimator:
# 
# <code>'scale'</code>
# 
# <code>'polynomial'</code>
# 
# <code>'model'</code>
# 
# The second element in the tuple  contains the model constructor
# 
# <code>StandardScaler()</code>
# 
# <code>PolynomialFeatures(include_bias=False)</code>
# 
# <code>LinearRegression()</code>

# In[53]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# ### Question 8
# 
# Use the list to create a pipeline object to predict the 'price', fit the object using the features in the list <code>features</code>, and calculate the R^2.

# In[59]:


from sklearn.metrics import mean_squared_error, r2_score


# In[60]:


pipe = Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
print(ypipe)


# In[61]:


print(r2_score(Y, ypipe))


# # Module 5: Model Evaluation and Refinement
# 
# Import the necessary modules:

# In[63]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# We will split the data into training and testing sets:

# In[64]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# ### Question 9
# 
# Create and fit a Ridge regression object using the training data, set the regularization parameter to 0.1, and calculate the R^2 using the test data. Take a screenshot of your code and the value of the R^2. You will need to submit it for the final project.

# In[65]:


from sklearn.linear_model import Ridge


# In[69]:


RidgeModel=Ridge(alpha = 0.1)
RidgeModel.fit(x_train, y_train)
print(RidgeModel.score(x_test, y_test))


# In[68]:


yhat = RidgeModel.predict(x_test)
print(yhat)


# ### Question 10
# 
# Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, and calculate the R^2 utilising the test data provided.

# In[71]:


pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
print(RidgeModel.score(x_test_pr, y_test))


# In[72]:


y_hat_pr = RidgeModel.predict(x_test_pr)
print(y_hat_pr)


# ### Author

# Tumedi Madihlaba

# In[ ]:




