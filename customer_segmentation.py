#!/usr/bin/env python
# coding: utf-8

# In[1]:


# @title import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[2]:


#@title load the dataset
df = pd.read_csv('train-set.csv')
df.head()


# In[3]:


#@ title Check for duplicates
print('Shape of dataset:', df.shape)
print('Number of duplicates', df['CustomerID'].duplicated().sum())


# In[4]:


#@title structure of dataset
df.info()


# In[5]:


#@title check for missing values
df.isna().sum()


# In[6]:


#@title validate dataset
df['CustomerID'].nunique()


# # **Exploratory Data Analysis**

# In[7]:


#@title Change column headers to lower case
df.columns = df.columns.str.lower()


# In[8]:


#@title Explore gender
nGender = df['gender'].nunique()
g = df['gender'].unique()
cGender = df.gender.value_counts()

print('The gender variable has {} unique categories and they are" {}'.format(nGender, g))
print('')
print(cGender)


# In[9]:


#@title visualize the distribution of gender
cGender.plot.pie(autopct = '%1.1f%%')

# label chart
plt.title('Distribution of Gender');


# In[10]:


#@title Explore marriage status
nMarried = df['married'].nunique()
m = df['married'].unique()
cMarried = df.married.value_counts()

print('The married variable has {} unique categories and they are" {}'.format(nMarried, m))
print('')
print(cMarried)


# In[11]:


cMarried.plot.pie(autopct = '%1.1f%%', colors = ['green', 'red'])

plt.title('Distribution of Mariage Status');


# In[12]:


df.columns


# In[13]:


num_cols = [
    'age', 'workexperience', 'familysize'
]
cat_cols = [
    'gender', 'married', 'graduated', 'profession', 'spendingscore'
]


# In[14]:


#@title explore the numeric variables in the dataset
sns.pairplot(df.loc[:, num_cols]);


# In[15]:


#@title explore the categorical variables in the dataset
for v in cat_cols:
  plt.figure(figsize=(10, 5))
  sns.countplot(x = v, data= df.loc[:, cat_cols])
  plt.title('Distribution of {}'.format(v));


# # **Data Preprocessing**

# In[16]:


#@title transform the binary categorical variables
df['gender'] = df['gender'].apply(lambda x:1 if x == "Female" else 2)


# In[17]:


#@title transform the marriage status variable

# get the mode
mode = df['married'].mode()[0]
# impute missing values with the mode
df['married'].fillna(mode, inplace = True)
df['married'].isna().sum()

# encode the values
df['married'] = df['married'].apply(lambda x:1 if x == 'Yes' else 0)
df['married'].head()


# In[18]:


#@title transform the graduation status variable

# get the mode
mode = df['graduated'].mode()[0]
# impute missing values with the mode
df['graduated'].fillna(mode, inplace = True)
df['graduated'].isna().sum()

# encode the values
df['graduated'] = df['graduated'].apply(lambda x:1 if x == 'Yes' else 0)
df['graduated'].head()


# In[19]:


#@title preprocess profession
mode = df['profession'].mode()[0]
df['profession'].fillna(mode, inplace = True)
df['profession'] = df['profession'].replace({
    'Healthcare': 1, 'Engineer':2, 'Lawyer':3, 'Entertainment':4,
    'Artist':5,'Executive':6, 'Doctor':7, 'Homemaker':8, 'Marketing':9
})
df['profession'].head()


# In[20]:


#@title impute missing values in work experience
median = df['workexperience'].median()
df['workexperience'].fillna(median, inplace = True)
df['workexperience'].isna().sum()


# In[21]:


#@title impute missing values in family size
mean = df['familysize'].mean()
median = df['familysize'].median()

df['familysize'].hist()
plt.axvline(mean, color = 'r', linewidth = 2, linestyle = '--', label = 'mean')
plt.axvline(median, color = 'g', linewidth = 2, linestyle = '-', label = 'median')
plt.legend();


# In[22]:


df['familysize'].fillna(median, inplace = True)
df['familysize'].isna().sum()


# In[23]:


df['spendingscore'].unique()


# In[24]:


#@title Encode spending score
df['spendingscore'] = df['spendingscore'].replace({
    'Low': 1,
    'Average': 2,
    'High': 3
})
df['spendingscore'].head()


# In[25]:


df.head()


# In[26]:


#@title Prepare dataset for model training
X = df.drop(columns=['customerid', 'category', 'segmentation'], axis=1)
X.head()


# # Choose the right number of Clusters

# ### The Within Cluster Sum of Squares (WCSS)

# In[27]:


wcss = []
for k in range(2, 11):
    km = KMeans(n_clusters = k, n_init = 25, random_state = 1234)
    km.fit(X)
    wcss.append(km.inertia_)

wcss_series = pd.Series(wcss, index = range(2, 11))

plt.figure(figsize=(8, 6))
ax = sns.lineplot(y = wcss_series, x = wcss_series.index)
ax = sns.scatterplot(y = wcss_series, x = wcss_series.index, s = 150)
ax = ax.set(xlabel = 'Number of Clusters (k)', 
            ylabel = 'Within Cluster Sum of Squares (WCSS)')


# ### The Average Silhouette Score

# In[28]:


from sklearn.metrics import silhouette_score

silhouette = []
for k in range(2, 11):
    km = KMeans(n_clusters = k, n_init = 25, random_state = 1234)
    km.fit(X)
    silhouette.append(silhouette_score(X, km.labels_))

silhouette_series = pd.Series(silhouette, index = range(2, 11))

plt.figure(figsize=(8, 6))
ax = sns.lineplot(y = silhouette_series, x = silhouette_series.index)
ax = sns.scatterplot(y = silhouette_series, x = silhouette_series.index, s = 150)
ax = ax.set(xlabel = 'Number of Clusters (k)', 
            ylabel = 'Average Silhouette Score')


# ### The Calinski Harabasz Score

# In[29]:


from sklearn.metrics import calinski_harabasz_score

calinski = []
for k in range(2, 11):
    km = KMeans(n_clusters = k, n_init = 25, random_state = 1234)
    km.fit(X)
    calinski.append(calinski_harabasz_score(X, km.labels_))

calinski_series = pd.Series(calinski, index = range(2, 11))

plt.figure(figsize=(8, 6))
ax = sns.lineplot(y = calinski_series, x = calinski_series.index)
ax = sns.scatterplot(y = calinski_series, x = calinski_series.index, s = 150)
ax = ax.set(xlabel = 'Number of Clusters (k)', 
            ylabel = 'Calinski Harabasz Score')


# ##### From our three graphs, we can conlude that our number of clusters should be 3

# In[30]:


X.shape


# In[31]:


#@title Model Building
model = KMeans(n_clusters=3, n_init=5, random_state=42)

# train model

y_kmeans=model.fit_predict(X)


# In[32]:


#@title make predictions
X['cluster'] = y_kmeans
X.head()


# In[33]:


X['cluster'].unique()


# In[34]:


#@title visualize clusters (with TSNE)
tsne = TSNE(n_components=2, verbose=1)
embedding = tsne.fit_transform(X)


# In[35]:


embedding


# In[36]:


#@title Transform the decomposed variables into a dataframe
df_decomposed = pd.DataFrame(columns=['x', 'y'], data = embedding)
df_decomposed.head()


# In[37]:


df_decomposed['cluster'] = X['cluster']
df_decomposed.head()


# In[38]:


df.head()


# In[39]:


df_decomposed['customerid'] = df['customerid']
df_decomposed.head()


# In[40]:


id = [467371, 463783]
df[df['customerid'].isin(id)]


# In[41]:


#@title visualize clusters
import plotly.express as px

fig = px.scatter(
    df_decomposed, x = 'x', y = 'y', color = 'cluster', hover_data = ['x', 'y', 'cluster', 'customerid']
)
fig.show()


# In[42]:


#@title PCA
pca = PCA(n_components=2)
decom = pca.fit_transform(X)
decom


# In[43]:


df_decomposed = pd.DataFrame(columns=['x', 'y'], data = decom)
df_decomposed.head()


# In[44]:


df_decomposed['cluster'] = X['cluster']
df_decomposed['customerid'] = df['customerid']
df_decomposed.head()


# In[45]:


#@title visualize clusters
import plotly.express as px

fig = px.scatter(
    df_decomposed, x = 'x', y = 'y', color = 'cluster', hover_data = ['x', 'y', 'cluster', 'customerid']
)
fig.show()

