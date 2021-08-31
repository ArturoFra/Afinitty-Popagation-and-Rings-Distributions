#!/usr/bin/env python
# coding: utf-8

# # Distribuciones en forma de anillos

# In[5]:


from math import sin, cos, radians, pi, sqrt
import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


def ring(r_min=0, r_max=1, n_samples=360):
    angle=rnd.uniform(0, 2*pi, n_samples)
    distance=rnd.uniform(r_min, r_max, n_samples)
    data=[]
    for a, d in zip(angle, distance):
        data.append([d*cos(a), d*sin(a)])
    return np.array(data)


# In[10]:


data1 = ring(3,5)
data2= ring(24, 27)

data = np.concatenate([data1, data2], axis=0)
labels = np.concatenate([[0 for i in range(0, len(data1))], [1 for i in range(0, len(data2))]])
plt.scatter(data[:,0], data[:,1], c=labels, s=5, cmap="autumn")


# # Algoritmo con Kmeans

# In[11]:


from sklearn.cluster import KMeans


# In[12]:


km = KMeans(2).fit(data)


# In[14]:


clust = km.predict(data)


# In[16]:


plt.scatter(data[:,0], data[:,1], c=clust, s=5, cmap="summer")


# # Algoritmo de los K medoids

# In[19]:


from pyclust import KMedoids


# In[20]:


kmed=KMedoids(2).fit_predict(data)


# In[21]:


plt.scatter(data[:,0], data[:,1], c=kmed, s=5, cmap="winter")


# # Algoritmo del clustering espectral

# In[22]:


from sklearn.cluster import SpectralClustering


# In[23]:


clust = SpectralClustering(2).fit_predict(data)


# In[24]:


plt.scatter(data[:,0], data[:,1], c=clust, s=5, cmap="autumn")


# In[25]:


#Podemos estimar la k:
 #* NO:Propagación de la afinidad
 #  Si: Podemos usar la distancia euclidea:
        #SI: USamos KMeans
        #NO: Buscar valores centrales
            #SI:K-Medoids
            #NO:Los datos son linealmente separables
                #Si: Clustering aglomerativo
                #No: Clustering Spectral
                


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




