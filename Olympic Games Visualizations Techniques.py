#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[15]:


import os
for dirname, _, filenames in os.walk("C:/Users/ADARSHA KUMAR BEHERA/Downloads/archive (15)/olympic_games.csv"):
    for filenmae in filenames:
        print(os.path.join(dirname, filename))


# In[16]:


olympics_data = pd.read_csv("C:/Users/ADARSHA KUMAR BEHERA/Downloads/archive (15)/olympic_games.csv")


# In[17]:


olympics_data.head(3)


# In[18]:


olympics_data.tail(3)


# In[19]:


olympics_data.isna().sum()


# In[20]:


olympics_data.columns


# In[35]:


### Most hosted Country of All Olympic Games is as follows:-

Max_count_host_1 = olympics_data['host_country'].value_counts()[:5].sort_values(ascending = False)
Max_count_host_1


# In[36]:


Max_count_host_1


# In[37]:


### Most hosted city of All Olympic Games is as follows:-

Max_count_host_2 = olympics_data['host_city'].value_counts()[:5].sort_values(ascending = False)
Max_count_host_2


# In[38]:


Max_count_host_2


# In[29]:


## Most Hosted Country for Olympics :-


# In[48]:


plt.figure(figsize = (8,8))

plt.pie(x = Max_count_host_1.values, labels = Max_count_host_1.index, autopct = '%1.1f%%')

plt.title("Top 5 most hosted Countries", fontweight = 'bold')
plt.tight_layout()
plt.show()


# In[31]:


## Most Hosted City for Olympics :-


# In[39]:


plt.figure(figsize = (9,8))

plt.pie(x = Max_count_host_2.values, labels = Max_count_host_2.index, autopct = "%1.1f%%")

plt.title("Top 5 hosted City of Olympics Games", fontweight = 'bold')
plt.tight_layout()
plt.show()


# In[49]:


### Visualizations of All Countries and Cities :-


# In[50]:


def bar_plot(_x,_y,xlabel,ylabel,title):
    plt.figure(figsize = (12,8))
    sns.set(style="whitegrid")  

    sns.barplot(x=_x, y=_y, palette="crest")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(rotation=90)


# In[54]:


olympics_data.columns


# In[55]:


Total_countries_host = olympics_data["host_country"].value_counts().sort_values(ascending = False)


# In[56]:


bar_plot(Total_countries_host.index,Total_countries_host.values,xlabel = "Countries", ylabel = "Games", 
         title = "All Country hosted the Olympics" )


# In[44]:


olympics_data.columns


# In[57]:


Total_olm_cities = olympics_data['host_city'].value_counts().sort_values(ascending = False)


# In[58]:


bar_plot(Total_olm_cities.index,Total_olm_cities.values,xlabel = "Cities", ylabel = "Games", 
         title = "Total followups cities hosted the Olympics")


# In[ ]:




