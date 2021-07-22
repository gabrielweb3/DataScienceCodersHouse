#!/usr/bin/env python
# coding: utf-8

# # Afterclass Clase 11

# ### Poner a grabar

# - Estandarización

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[36]:


import numpy as np

x = np.random.rand(1000)


# In[38]:


plt.hist(x)


# In[42]:


y = (x-np.mean(x))/np.std(x)
plt.hist(y)


# https://heartbeat.fritz.ai/understanding-the-mathematics-behind-principal-component-analysis-efd7c9ff0bb3

# # PCA

# ![image.png](attachment:image.png)

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
plt.style.use('ggplot')

# Load the data
df = sns.load_dataset('iris')


# In[44]:


df


# In[46]:


X = df.drop(columns='species')
y = df['species']


# In[49]:


X


# # Escalamos

# In[50]:


# Z-score the features
scaler = StandardScaler()
scaler.fit(X)


# In[ ]:


X_scale = scaler.transform(X)


# In[53]:


X_scale


# In[56]:


plt.scatter(X['sepal_length'], X['sepal_width'])


# In[57]:


plt.scatter(X_scale[:,0], X_scale[:,1])


# # Hacemos PCA

# In[ ]:


# The PCA model
pca = PCA() # estimate only 2 PCs
X_new = pca.fit_transform(X_scale) # project the original data into the PCA space


# In[59]:


X_new.shape


# In[60]:


X.columns


# In[63]:


df_pca = pd.DataFrame(X_new, columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4'])


# In[64]:


df_pca


# In[66]:


df['speciesn'] = df.species.map( {'setosa':0 , 'versicolor':1, 'virginica':2} )


# In[67]:


df.species


# In[74]:


fig, axes = plt.subplots(1,3,figsize=(15,4))
axes[0].scatter(X_scale[:,0], X_scale[:,1], c=df.speciesn.values)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Antes de PCA')
axes[0].set_xlim(-3,3)
axes[0].set_ylim(-3,3)

axes[1].scatter(X_new[:,0], X_new[:,1], c=df.speciesn.values)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Despues de  PCA')
axes[1].set_xlim(-3,3)
axes[1].set_ylim(-3,3)


axes[2].scatter(X_new[:,2], X_new[:,3], c=df.speciesn.values)
axes[2].set_xlabel('PC3')
axes[2].set_ylabel('PC4')
axes[2].set_title('Despues de  PCA')
axes[2].set_xlim(-3,3)
axes[2].set_ylim(-3,3)

plt.show()


# In[75]:


pca.explained_variance_ratio_


# In[76]:


pca.components_,


# In[14]:


df_components = pd.DataFrame(pca.components_,columns=df.iloc[:,:4].columns,index = ['PC-1','PC-2','PC-3','PC-4'])
df_components


# In[ ]:





# In[31]:


fig, axes = plt.subplots(1,1,figsize=(6,6))

axes.scatter(X_new[:,0], X_new[:,1], c=df.speciesn.values)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_title('Despues de  PCA')
coeff = np.transpose(pca.components_[0:2, :])
n = coeff.shape[0]
for i in range(n):
        plt.arrow(0, 0, coeff[i,0]*3, coeff[i,1]*3,color='r',alpha=0.5) 
        #if labels is None:
        plt.text(coeff[i,0]* 3.15, coeff[i,1] * 3.15, X.columns[i], color='g', ha='center', va='center')
        #else:
            #plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color='g', ha='center', va='center')


# In[29]:


pca.components_.shape


# In[19]:


X.columns[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




