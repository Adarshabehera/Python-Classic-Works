#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Tata_Motors_Employee_Reviews_from_AmbitionBox.csv")


# In[4]:


df.head(3)


# In[5]:


df.tail(3)


# In[6]:


df.shape


# In[9]:


df.duplicated()


# In[10]:


df.isna().sum()


# In[16]:


df["work_life_balance"].nunique()


# ## For creating the subplots :-

# In[17]:


df.columns


# In[18]:


df.head(2)


# In[27]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
# Plot Overall Rating distribution
sns.histplot(df['Overall_rating'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Overall Rating Distribution')

# Plot Work Life Balance distribution
sns.histplot(df['work_satisfaction'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Work Life Balance Distribution')

# Plot Skill Development distribution
sns.histplot(df['skill_development'], kde=True, ax=axes[0, 2])
axes[0, 2].set_title('Skill Development Distribution')

# Plot Salary and Benefits distribution
sns.histplot(df['salary_and_benefits'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Salary and Benefits Distribution')

# Plot Job Security distribution
sns.histplot(df['job_security'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Job Security Distribution')

# Plot Career Growth distribution
sns.histplot(df['career_growth'], kde=True, ax=axes[1, 2])
axes[1, 2].set_title('Career Growth Distribution')


plt.tight_layout()

plt.show()


# ### from above we specify that each terms are Left Skewed and mean < median which is not satisfactory and applicable.

# In[28]:


df.columns


# In[34]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 10))


# Plot Work Life Balance vs. Overall Rating
sns.boxplot(data=df, x='work_life_balance', y='Overall_rating', ax=axes[0])
axes[0].set_title('Work Life Balance vs. Overall Rating')

# Plot Salary and Benefits vs. Overall Rating
sns.boxplot(data=df, x='salary_and_benefits', y='Overall_rating',  ax=axes[1])
axes[1].set_title('Salary and Benefits vs. Overall Rating')

# Plot Work Satisfaction vs. Overall Rating
sns.boxplot(data=df, x='work_satisfaction', y='Overall_rating', ax=axes[2])
axes[2].set_title('Work Satisfaction vs. Overall Rating')

# Adjust layout
plt.tight_layout()

# Show plots
plt.show()


# ### Preprocess the Nan value in each and Every areas of data i.e, (rows & Columns):-

# In[35]:


from sklearn.impute import SimpleImputer

s1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
s2 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')


# In[36]:


df.columns


# In[42]:


df["work_satisfaction"].fillna(value = df["work_satisfaction"].mean(),inplace = True)
df["skill_development"].fillna(value = df["skill_development"].mean(),inplace = True)


# In[43]:


df.isna().sum()


# In[41]:


df.head(3)


# In[44]:


df = df.drop(["Likes","Dislikes"], axis = 1)


# In[46]:


y = df["work_satisfaction"]
df = df.drop(["work_satisfaction"], axis = 1)


# In[47]:


df.head(3)


# In[48]:


cat_cols = df.select_dtypes(include = 'object').columns
num_cols = df.select_dtypes(exclude = 'object').columns


# In[49]:


cat_cols


# In[50]:


num_cols


# In[51]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
numerical_transformer = SimpleImputer(strategy = 'constant')


# ### Preprocess the categorical data:-

# In[52]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown = 'ignore'))
])


# In[54]:


preprocessor = ColumnTransformer(
   transformers = [
       ('num',numerical_transformer, num_cols),
       ('cat',categorical_transformer, cat_cols)
   ])


# In[55]:


X = df


# In[57]:


X.head(3)


# In[62]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[63]:


import catboost as cb
my_pipeline = Pipeline(steps = [
    ('preprocessor',preprocessor),
    ('model',cb.CatBoostRegressor(verbose = 0))
])


# In[64]:


my_pipeline.fit(X_train,y_train)


# In[65]:


my_pipeline.score(X_test, y_test)


# In[66]:


import xgboost as xg
my_pipeline2 = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', xg.XGBRegressor(n_estimators = 500))
])


# In[67]:


my_pipeline2.fit(X_train,y_train)


# In[69]:


my_pipeline2.score(X_test, y_test)


# ### From above two modelling , we analysed that ensembling one id giving the best accuracy of 61 % as compared to XG-boost modelling
# 

# In[ ]:




