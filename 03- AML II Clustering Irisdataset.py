#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score


# In[21]:


#cluster the different flowers that we have. 


# In[22]:


data = load_iris()


# # **Data Frame**

# In[23]:


df = pd.DataFrame(data=data.data, columns=data.feature_names)
df.head()


# In[24]:


df["Species"] = data.target
df["Species_Name"] = df["Species"]
df.sample(10)


# **Adding the name of each Species_Name**

# In[25]:


for index, row in df.iterrows():
    species = int(row["Species"])
    df.loc[index, "Species_Name"] = data.target_names[species]

df.sample(10)


# In[26]:


df.shape


# # **K- Means Clustering** 

# PASS JUST THE FEATURES TO KMEANS. (4 features)
# NO SPECIES AND SPECIES NAME.

# **Split**

# In[27]:


X = df[data.feature_names]
X.head()


# In[28]:


kmeans= KMeans(n_clusters=3, random_state=0, n_init="auto")
kmeans.fit(X)


# **Predict X and tell what are the labels:**

# In[29]:


predicted_clusters = kmeans.predict(X)
df["Cluster"] = predicted_clusters
df.sample(10)


# In[30]:


'''centroids = kmeans.cluster_centers_
centroids
'''


# # **Scatter Plot**

# **ORIGINAL DATA**

# In[31]:


fig = px.scatter_3d(df,
                    x="petal length (cm)",
                    y="petal width (cm)",
                    z="sepal length (cm)",
                    color="Species_Name",
                    height=800)
fig.show()


# **PLOTTING CLUSTERS:**

# In[32]:


fig = px.scatter_3d(df,
                    x="petal length (cm)",
                    y="petal width (cm)",
                    z="sepal length (cm)",
                    color="Cluster",
                    height=800)
fig.show()


# # **MAPPING CLUSTERS:**

# In[33]:


df["Cluster_Name"] = ''
df["Cluster_Name"] = np.where(df["Cluster"] == 0, 'setosa', df["Cluster_Name"])
df["Cluster_Name"] = np.where(df["Cluster"] == 1, 'virginica', df["Cluster_Name"])
df["Cluster_Name"] = np.where(df["Cluster"] == 2, 'versicolor', df["Cluster_Name"])
df.head(25)


# # **Metrics:**

# **CONFUSION MATRIX:**

# In[34]:


cm = confusion_matrix(df["Species_Name"], df["Cluster_Name"], labels=data.target_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot()
plt.show()


# In[35]:


#0 setosa 100% accurate 50/50
#versicolor 47 correctly, WRONG: 3 virginica and 1
#virginica: 


# In[ ]:


# Recommender system: personal 
# Group recomendation: 
#also bought: association rule 

