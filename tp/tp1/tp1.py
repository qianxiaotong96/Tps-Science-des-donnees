#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;background-color:#336699;color:white;">TP1
# Introduction au module scikit-learn</h1>
# <h3 style="color:#8080C0">A. Importation des libraires</h3>
# <h4>1- Importez scikit-learn :
# Dans Python, la plupart des fonctions sont incluses dans des librairies, qu’il faut importer pour pourvoir les utiliser. Par exemple pour importer scikit-learn il faut écrire :
# from sklearn import *</h4>

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: QIAN Xiaotong
@author: BIAN Yiping
"""
from sklearn import *


# <h4>2- Importez les librairies numpy (calcul scientifique) et matplotlib.pyplot (figures).</h4>

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# <h3 style="color:#8080C0">B. Manipulation d’un jeu de données</h3>
# <h4>1- Chargez les données Iris avec la commande :
# iris = datasets.load_iris()
# La variable iris est un objet qui contient la matrice des données (iris.data), un vecteur de numéro de classe (target), ainsi que les noms des variables (feature_names) et le nom des classes (target_names).</h4>

# In[3]:


iris = datasets.load_iris()


# <h4>2- Affichez les données, les noms des variables et le nom des classes (utilisez print).</h4>

# In[4]:


print("les données : ",iris.data) 
#print("un vecteur de numéro de class : ",iris.target) 
print("les noms des variable : ",iris.feature_names) 
print("les noms de classe",iris.target_names) 


# <h4>3- Affichez le nom des classes pour chaque donnée</h4>

# In[5]:


j=0
for i in iris.target:
    print("la donnée numéro ",j," : ",iris.data[j],',',iris.target_names[i])
    j=j+1


# <h4>4- Affichez la moyenne (mean), l’ecart-type (std), le min et le max pour chaque variable.</h4>

# In[6]:


print("la moyenne de chaque variable : ",iris.data.mean(axis=0))
print("l'écart-type de chaque variable : ",iris.data.std(axis=0))
print("le minimum de chaque variable : ",iris.data.min(axis=0))
print("le maximum de chaque variable : ",iris.data.max(axis=0))


# <h4>5- En utilisant les attributs size et shape, affichez le nombre de données, le nombre de
# variables et le nombre de classes.</h4>

# In[7]:


print("le nombre de donnees = ",iris.data.shape[0])
print("le nombre de variable = ",iris.data.shape[1])
print("le nombre de class = ",iris.target_names.size)


# <h3 style="color:#8080C0">C. Téléchargement et importation de données</h3>
# <p style="color:orange">Cette exo prend du temps pour exécuter, donc je met des commentaires, pour tester,il faut enlever les commentaires<p>
# <h4>1- Importez les données 'MNIST original'.</h4>

# In[1]:


'''
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784',version=1)
'''


# <h4>2- Affichez la matrice des données, le nombre de données et de variables, les numéros de classes pour chaque donnée, ainsi que la moyenne, l’écart type, les valeurs min et max pour chaque variable ; enfin donnez le nombre de classe avec la fonction unique.</h4>

# In[2]:


'''
print("la matrice des données : ",mnist.data)
print("le nombre de données : ",mnist.data.shape[0])
print("le nombre de variables : ",mnist.data.shape[1])
#print("un vecteur de numero de class : ",mnist.target)  


print("les numéros de classes pour chaque donnée:")

j=0
for i in mnist.target:
    print("les données numéro ",j," : ",mnist.data[j],','+i)
    j=j+1
'''


# In[3]:


'''
print("la moyenne de chaque variable : ",mnist.data.mean(axis=0))
print("l'écart-type de chaque variable : ",mnist.data.std(axis=0))
print("le minimum de chaque variable : ",mnist.data.min(axis=0))
print("le maximum de chaque variable : ",mnist.data.max(axis=0))
print("le nombre de class : ",len(np.unique(mnist.target)))
'''


# <h3 style="color:#8080C0">D. Génération de données et affichage</h3>
# <h4>1. Utiliser l’aide (help) pour voir comment utiliser la fonction datasets.make_blobs.</h4>

# In[12]:


from sklearn.datasets import make_blobs


# <p style = "color:green">
# La fonction make_blobs consiste à générer un ensemble de données pour le clustering</p>
# <h4>2. Générez 1000 données de deux variables réparties en 4 groupes.</h4>

# In[13]:


data,label = make_blobs(n_samples=1000,n_features=2,centers=4)


# <h4>3. Utilisez les fonctions figure, scatter, title, xlim, ylim, xlabel, ylabel et show pour afficher les données avec des couleurs correspondant aux classes. Les axes x et y seront dans l’intervalle [-15, 15] et devrons avoir un titre. La figure doit aussi avoir un titre.</h4>

# In[14]:


plt.figure(figsize=(5,5))
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.xlabel('x')
plt.ylabel('y')
plt.title('graphe intéressant')
plt.scatter(data[:, 0], data[:, 1], c=label)


# <h4>4. Générez 100 données de deux variables réparties en 2 groupes, puis 500 données de deux variables réparties en 3 groupes. Concaténez (vstack et hstack) les deux jeux de données et les numéros de classe pour fusionner les deux jeux de données. Affichez les trois ensembles avec scatter comme précédemment.</h4>

# In[15]:


data1,label1 = make_blobs(n_samples=100,n_features=2,centers=2)
plt.figure(figsize=(5,5))
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.xlabel('x1')
plt.ylabel('y1')
plt.title('graphe intéressant 1 avec 100 données')
plt.scatter(data1[:, 0], data1[:, 1], c=label1)
plt.show()
data2,label2 = make_blobs(n_samples=500,n_features=2,centers=3)
label2[label2==0]=3
label2[label2==1]=4
label2[label2==2]=5
plt.figure(figsize=(5,5))
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.xlabel('x2')
plt.ylabel('y2')
plt.title('graphe intéressant 2 avec 500 données')
plt.scatter(data2[:, 0], data2[:, 1], c=label2)


# In[16]:


data3=np.vstack((data1,data2))
label3=np.hstack((label1,label2))
plt.figure(figsize=(5,5))
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.xlabel('x3')
plt.ylabel('y3')
plt.title('graphe intéressant 3 avec 600 données')
plt.scatter(data3[:, 0], data3[:, 1], c=label3)


# In[ ]:




