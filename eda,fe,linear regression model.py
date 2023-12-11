#!/usr/bin/env python
# coding: utf-8

# # Zomato Dataset Explorative Data Analysis

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df=pd.read_csv('zomato.csv',encoding='latin-1')
df.head()


# In[3]:


df.info()


# In[4]:


df.index


# In[5]:


df.columns


# In[6]:


df.values


# In[7]:


df.dtypes


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


df['Country Code'].unique()


# In[11]:


df.size


# In[12]:


import seaborn as sns


# In[13]:


df=sns.load_dataset('healthexp')


# In[14]:


df.head()


# In[15]:


import numpy as np


# In[16]:


df.cov()


# In[17]:


df.corr(method='spearman')


# In[18]:


df=pd.read_csv('zomato.csv',encoding='latin-1')
df.head()


# In[19]:


df.info()


# In[20]:


df.isnull().sum()


# In[21]:


df.notna().size


# In[22]:


df.isnull().size


# In[23]:


df['Cuisines'].unique()


# In[24]:


df.dropna(inplace=True)


# In[25]:


df.isnull().sum()


# In[26]:


df.index


# In[27]:


df.info()


# In[28]:


df.describe()


# In[29]:


df['Restaurant Name'].unique()


# In[30]:


from sklearn.preprocessing import LabelEncoder


# In[31]:


labelencoder=LabelEncoder()


# In[32]:


df['Restaurant Name']=labelencoder.fit_transform(df['Restaurant Name'])


# In[33]:


df.info()


# In[34]:


df.head()


# In[35]:


df['Restaurant Name'].max()


# In[36]:


df['City']=labelencoder.fit_transform(df['City'])
df['Currency']=labelencoder.fit_transform(df['Currency'])


# In[37]:


df.reset_index(inplace=True)


# In[38]:


df1=pd.get_dummies(df['Has Table booking'],drop_first=True,prefix='Has Table booking')


# In[39]:


df2=pd.get_dummies(df['Has Online delivery'],drop_first=True,prefix='Has Online delivery')
df3=pd.get_dummies(df['Is delivering now'],drop_first=True,prefix='Is delivering now')


# In[40]:


df4=pd.get_dummies(df['Switch to order menu'],prefix='Switch to order menu')
df['Switch to order menu'].unique()


# In[41]:


df=pd.concat([df,df1],axis=1)
df=pd.concat([df,df2],axis=1)
df=pd.concat([df,df3],axis=1)
df=pd.concat([df,df4],axis=1)


# In[42]:


df.info()


# In[43]:


df.tail(1)


# In[44]:


df.drop('index',axis=1,inplace=True)
df.drop('Has Table booking',axis=1,inplace=True)
df.drop('Has Online delivery',axis=1,inplace=True)
df.drop('Is delivering now',axis=1,inplace=True)
df.drop('Switch to order menu',axis=1,inplace=True)


# In[45]:


df.info()


# In[46]:


df['Rating text'].unique()


# In[47]:


df['Rating text']=df['Rating text'].map({'Excellent':'5', 'Very Good':'4', 'Good':'3', 'Average':'2', 'Not rated':'0', 'Poor':1})


# In[48]:


df.info()


# In[49]:


df[df['Rating text']=='0']


# In[50]:


df['Rating text']=df['Rating text'].astype(int)


# In[51]:


df.info()


# In[52]:


df.head()


# In[53]:


[features for features in df.columns if df[features].isnull().sum()>0]


# In[55]:


df1=pd.read_csv('booking_hotel.csv',encoding='latin1')


# In[56]:


plt.figure(figsize=[5,8])
sns.heatmap(df.isnull())


# In[57]:


df.dtypes


# In[58]:


df['Rating color'].value_counts()


# In[59]:


df['Rating color'].unique()


# In[60]:


plt.pie(df['Rating color'].value_counts(),labels=df['Rating color'].value_counts().index,autopct='%1.2f%%')


# In[61]:


ratings=df.groupby(['Aggregate rating','Rating color','Rating text']).size().reset_index().rename(columns={0:'Rating Count'})


# In[62]:


ratings


# In[63]:


plt.figure(figsize=[10,12])
sns.barplot(x='Aggregate rating',y='Rating Count',hue='Rating color',data=ratings,palette=['blue','red','orange','yellow','green','green'])


# In[64]:


sns.countplot(x='Rating color',data=ratings,palette=['blue','red','orange','yellow','green','green'])


# # simple linear regression

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[66]:


df=pd.read_csv('height-weight.csv')
df.head()


# In[67]:


plt.figure(figsize=[9,9])
plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")
plt.title('regression')


# In[68]:


x=df[['Weight']]
y=df[['Height']]


# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[71]:


x.shape


# In[72]:


x_train.shape


# In[73]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[74]:


from sklearn.preprocessing import StandardScaler


# In[75]:


scaler=StandardScaler()


# In[76]:


x_train=scaler.fit_transform(x_train)


# In[77]:


x_test=scaler.transform(x_test)
plt.scatter(x_train,y_train)


# In[78]:


from sklearn.linear_model import LinearRegression


# In[79]:


regressor=LinearRegression()


# In[80]:


regressor.fit(x_train,y_train)


# In[81]:


regressor.coef_,regressor.intercept_


# In[82]:


regressor.intercept_


# In[83]:


plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train),'r')


# In[84]:


y_pred_test=regressor.predict(x_test)


# In[85]:


plt.scatter(x_test,y_test)
plt.plot(x_test,regressor.predict(x_test),'r')


# In[86]:


y_pred_test,y_test


# In[87]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[88]:


mse=mean_squared_error(y_test,y_pred_test)
mae=mean_absolute_error(y_test,y_pred_test)
rmse=np.sqrt(mse)
print(mse,mae,rmse)


# In[89]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred_test)


# In[90]:


score


# In[91]:


scaled_weight=scaler.transform([[80]])
scaled_weight


# In[92]:


regressor.predict([scaled_weight[0]])


# In[93]:


plt.scatter(y_test,y_pred_test)


# In[94]:


residuals=y_test-y_pred_test
residuals


# In[95]:


sns.distplot(residuals,kde=True)


# In[96]:


from sklearn.datasets import fetch_california_housing


# In[97]:


california=fetch_california_housing()


# In[98]:


california.keys()


# In[99]:


type(california)


# In[100]:


print(california.DESCR)


# In[101]:


california.target_names


# In[102]:


print(california.data)


# In[103]:


print(california.target)


# In[104]:


california.feature_names


# In[105]:


dataset=pd.DataFrame(california.data,columns=california.feature_names)


# In[106]:


dataset.head()


# In[107]:


dataset['Price']=california.target


# In[108]:


dataset.head()


# In[109]:


dataset.info()


# In[110]:


dataset.isnull().sum()


# In[111]:


dataset.describe()


# In[112]:


sns.heatmap(dataset.corr(),annot=True)


# In[113]:


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[114]:


x.head()


# In[115]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=10)


# In[116]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[117]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[118]:


x_train=scaler.fit_transform(x_train)


# In[119]:


x_test=scaler.transform(x_test)


# In[120]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression


# In[121]:


regression.fit(x_train,y_train)


# In[122]:


regression.coef_


# In[123]:


regression.intercept_


# In[124]:


y_pred=regression.predict(x_test)


# In[125]:


y_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




