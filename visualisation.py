#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv("zomato.csv",encoding="latin-1")
df.head()


# In[4]:


df.columns


# In[5]:


df.info()


# In[7]:


df.describe()


# ##in data analysis what all things we do
# 1.missing values
# 2.explore about the numerical values
# 3.explore about categorical variables
# 4.finding relationship between features

# In[8]:


df.isnull().sum()


# In[12]:


[features for features in df.columns if df[features].isnull().sum()>0]


# In[13]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[16]:


df_country=pd.read_excel('country-code.xlsx')
df.head()


# In[19]:


final_df=pd.merge(df,df_country,on='Country Code',how='left')


# In[20]:


final_df.head()


# In[21]:


#to check data types
final_df.dtypes


# In[22]:


final_df.columns


# In[25]:


country_names=final_df.Country.value_counts().index


# In[28]:


country_val=final_df.Country.value_counts().values


# In[36]:


##pie chart top 3 countries thatt uses zomato 
plt.pie(country_val[:3],labels=country_names[:3],autopct='%3f%%')


# observation: zomato maximum is in india then us then uk

# In[38]:


final_df.columns


# In[44]:


ratings=final_df.groupby(['Aggregate rating','Rating color','Rating text']).size().reset_index().rename(columns={0:'Rating Count'})


# In[45]:


ratings


# ## observation
# 1. when rating is between 4.5 to4.9 --> excellent
# 2. when rating are between 4.0 to 4.4-->very good
# 3. when rating are between 3.6 to 4.0--> good
# 4. average
# 5. poor
# 6. very poor

# In[47]:


ratings.head()


# In[51]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (12, 6)
sns.barplot(x="Aggregate rating",y="Rating Count",data =ratings)


# In[54]:


sns.barplot(x="Aggregate rating",y="Rating Count",hue='Rating color',data =ratings,palette=['white','red','orange','yellow','green','green'])


# observation
# 1. not rated count is very high
# 2. maximum number of rating are between 2.5 to 3.4

# In[56]:


## count plot
sns.countplot(x="Rating color",data=ratings,palette=['white','red','orange','yellow','green','green'])


# In[57]:


## find the countries name that has given 0 rating


# In[61]:


final_df.columns


# In[62]:


df2=final_df["Aggregate rating"]


# In[66]:


df_2=pd.DataFrame(df2)


# In[67]:


df_2.head()


# In[68]:


df_2['country']=final_df['Country']


# In[81]:


df_2.head()


# In[82]:


df_2.dtypes


# In[87]:


for i,j in df_2.iterrows():
    if j['Aggregate rating']==0:
        print(i,j)


# In[91]:


final_df.groupby(['Aggregate rating','Country']).size().reset_index()


# observation 
# maximum number of 0 ratings are from indian customers
# 

# In[92]:


# find out which country used which currency


# In[93]:


final_df.head()


# In[103]:


final_df.groupby(["Country","Currency"]).size().reset_index()


# In[104]:


#which country do have online deliveries option


# In[106]:


final_df.columns


# In[113]:


final_df.loc[final_df["Has Online delivery"]=="Yes"].groupby(["Country"]).size().reset_index()


# In[115]:


final_df[["Has Online delivery","Country"]].groupby(["Has Online delivery","Country"]).size().reset_index()


# ## observation
# 1. online deliveries are available in India and UAE

# In[116]:


## create a pie chart for cities distribution


# In[122]:


final_df.columns


# In[124]:


final_df.City.value_counts().index


# In[126]:


plt.pie(final_df.City.value_counts().values[:5],labels=final_df.City.value_counts().index[:5])


# In[ ]:




