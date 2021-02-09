#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;background-color:#336699;color:white;">TP2
# Prétraitement et visualisation de données</h1>
# <h3 style="color:#8080C0">A. Normalisation de données</h3>
# <h4>1- Créez la matrice X suivante : [[1, -1, 2], [2, 0, 0],
# [0, 1, -1]]</h4>

# In[16]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: QIAN Xiaotong
@author: BIAN Yiping
"""
import numpy as np
from sklearn import * 
import matplotlib.pyplot as plt

x = np.array([[1,-1,2],[2,0,0],[0,1,-1]]) 


# <h4>2- Visualisez X et calculez la moyenne et la variance de X.</h4>

# In[17]:


print("x = ",x)
moyen = np.mean(x)
print("moyen = ",moyen)
var = np.var(x)
print("variance = ",var)


# <h4>3- Utilisez la fonction scale pour normaliser la matrice X. Que constatez vous ?</h4>

# In[6]:


x_scaled = preprocessing.scale(x)
x_scaled


# <p style="color:green">Remarque : la somme de chaque colonne est égale à 0 </p>
# <h4>4- Calculer la moyenne et la variance de la matrice X normalisé. Expliquez le résultat obtenu.</h4>

# In[8]:


moyen = np.mean(x_scaled)
print("moyen = ",moyen)
var = np.var(x_scaled)
print("variance = ",var)


# <h3 style="color:#8080C0">B. Normalisation MinMax</h3>
# <h4>1- Créez la matrice X2 suivante : [[1, -1, 2], [2, 0, 0],
# [0, 1, -1]]</h4>

# In[15]:


x2 = np.array([[1,-1,2],[2,0,0],[0,1,-1]]) 


# <h4>2- Visualisez la matrice et calculez la moyenne sur les variables.</h4>

# In[14]:


print(x2)
moyen = np.mean(x2,axis=0)
print("moyen = ",moyen)


# <h4>3- Normalisez les données dans l’intervalle [0 1]. Visualisez les données normalisées et calculez la moyenne sur les variables. Que constatez-vous ?</h4>

# In[18]:


scale = preprocessing.MinMaxScaler(feature_range=(0,1))
x2_scaled = scale.fit_transform(x2)
print(x2_scaled)
moyen = np.mean(x2_scaled,axis=0)
print("moyen = ",moyen)


# <h3 style="color:#8080C0">C. visualisation de données</h3>
# <h4>1- Chargez les données Iris</h4>

# In[19]:


iris = datasets.load_iris()


# <h4>2- Visualisez le nuage de points en 2D avec des couleurs correspondant aux classes en utilisant toutes les combinaisons de variables. Quelle est la meilleure visualisation ? Justifiez votre réponse.</h4>

# In[20]:


print(iris.feature_names)
# on obtient ['sepal length (cm)', 
#             'sepal width (cm)', 
#             'petal length (cm)', 
#             'petal width (cm)']

# on a les combinaisons possibles suivantes:
# 'sepal length (cm)' * 'sepal width (cm)' => iris.data[:,0] * iris.data[:,1]
# 'sepal length (cm)' * 'petal length (cm)' => iris.data[:,0] * iris.data[:,2]
# 'sepal length (cm)' * 'petal width (cm)' => iris.data[:,0] * iris.data[:,3]
# 'sepal width (cm)' * 'petal length (cm)' => iris.data[:,1] * iris.data[:,2]
# 'sepal width (cm)' * 'petal width (cm)' => iris.data[:,1] * iris.data[:,3]
# 'petal length (cm)' * 'petal width (cm)' => iris.data[:,2] * iris.data[:,3]

label = iris.target
graphe = 1
for i in range(3):
    x = iris.data[:,i]
    for j in range(i+1,4):
        y = iris.data[:,j]
        x_min,x_max = x.min(),x.max()
        y_min,y_max = y.min(),y.max()

        plt.figure()
        string = "combinaison "+str(graphe)+" "+iris.feature_names[i]+" * "+iris.feature_names[j]
        plt.title(string)
        plt.scatter(x, y, c=label)
        plt.xlabel(iris.feature_names[i])
        plt.ylabel(iris.feature_names[j])
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        #plt.show()
        graphe = graphe +1
plt.show()


# <p style="color:green">Selon moi, la 6ème est mieux, car les 3 groups sont separe clairement</p>
# <h3 style="color:#8080C0">D. Réduction de dimensions et visualisation de données</h3>
# <h4>1- Les méthodes PCA et LDA peuvent etre importé à partir des package suivants :
# import from sklearn.decomposition import PCA from sklearn.lda import LDA
# </h4>

# In[21]:


from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# <h4>2- Analysez le manuel d’aide pour ces deux fonctions (pca et lda) et appliquez les sur la base Iris. Il faudra utiliser pca.fit(Iris).transform(Iris) et sauvegardez les résultats dans IrisPCA pour la PCA et IrisLDA pour la LDA.</h4>

# In[22]:


pca = PCA(n_components=2)
x_p = pca.fit(iris.data).transform(iris.data)
print(x_p)
x = x_p[:,0]
y = x_p[:,1]
x_min,x_max = x.min(),x.max()
y_min,y_max = y.min(),y.max()

plt.figure()
plt.title('apres la methode pca')
plt.scatter(x, y, c=label)
plt.xlabel('dimension 1')
plt.ylabel('diemnsion 2')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)


# <h4>3- Visualisez les nuages de points avec les nouvelles axes obtenus : une image pour l’ACP et une autre pour l’ADL et utiliser la classe de Iris comme couleurs de points. Quelle différence constatez-vous entre les deux visualisations? Expliquer votre raisonnement.</h4>

# In[23]:


lda = LDA(n_components=2)
x_l = lda.fit(iris.data,iris.target).transform(iris.data)
print(x_l)
x = x_l[:,0]
y = x_l[:,1]
x_min,x_max = x.min(),x.max()
y_min,y_max = y.min(),y.max()

plt.figure()
plt.title('apres la methode lda')
plt.scatter(x, y, c=label)
plt.xlabel('dimension 1')
plt.ylabel('diemnsion 2')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)


# In[ ]:




