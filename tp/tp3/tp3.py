#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;background-color:#336699;color:white;">TP3
# Introduction à la classification</h1>
# <h3 style="color:#8080C0">A. Plus Proche Voisin</h3>
# <h4>1. Créez une fonction PPV(X,Y) qui prend en entrée des données X et des étiquettes Y et qui renvoie une étiquette, pour chaque donnée, prédite à partir du plus proche voisin de cette donnée. Ici on prend chaque donnée, une par une, comme donnée de test et on considère toutes les autres comme données d’apprentissage. Cela nous permet de tester la puissance de notre algorithme selon une méthode de validation par validation croisée (cross validation) de type “leave one out”.</h4>

# In[3]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: QIAN Xiaotong
@author: BIAN Yiping
"""

from sklearn import * 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  
from scipy.stats import mode

'''
x -> les donnees
y -> class
renvoie la class du voisin qui est plus proche de cette donnees
'''

# print(metrics.pairwise.euclidean_distances(iris.data))  
# permet de calculer la distance de tous les voisin de chaque donnee
# np.argsort(metrics.pairwise.euclidean_distances(x))
# trier le tab de distance dans l'ordre croissant par indice.
# print(np.argsort(metrics.pairwise.euclidean_distances(iris.data))[:,1])
# renvoie la deuxième indice de la tab triée
# iris.target[np.argsort(metrics.pairwise.euclidean_distances(x))[:,1]]
# renvoie les class de chaque donnees de la tab qui contient la deuxième indice de la tab triée
def ppv(x,y):
    z = y[np.argsort(metrics.pairwise.euclidean_distances(x))[:,1]]
    return z 


# <h4>2. La fonction PPV calcule une étiquette prédite pour chaque donnée. Modifiez la fonction pour calculer et renvoyer l’erreur de prédiction : c’est à dire le pourcentage d’étiquettes mal prédites.</h4>

# In[4]:


def ppv_err(x,y):
    res = np.zeros(len(y))
    z = y[np.argsort(metrics.pairwise.euclidean_distances(x))[:,1]]
    res = len(np.nonzero(abs(y-z)))/len(y)
    return res


# <h4>3. Testez sur les données Iris.</h4>

# In[5]:


iris = datasets.load_iris()
z = ppv(iris.data,iris.target)

print(z)

err = ppv_err(iris.data,iris.target)
print(err)


# <h4>4. Testez la fonction des K Plus Proches Voisins de sklearn (avec ici K = 1). Les
# résultats sont-ils différents? Testez avec d’autres valeurs de K.</h4>

# In[6]:


neigh_1 = KNeighborsClassifier(n_neighbors=1)  
neigh_1.fit(iris.data, iris.target)
print(neigh_1.predict(iris.data))


# In[7]:


neigh_2 = KNeighborsClassifier(n_neighbors=2)  
neigh_2.fit(iris.data, iris.target)
print(neigh_2.predict(iris.data))


# In[8]:


neigh_3 = KNeighborsClassifier(n_neighbors=3)  
neigh_3.fit(iris.data, iris.target)
print(neigh_3.predict(iris.data))


# <h4>5. BONUS : Modifiez la fonction PPV pour qu’elle prenne en entrée un nombre K de voisins (au lieu de 1). La classe prédite sera alors la classe majoritaire parmi les K voisins.</h4>

# In[9]:


def ppv_k(x,y,k):
    data = np.argsort(metrics.pairwise.euclidean_distances(x))
    tab = np.zeros((k,len(data)))
    for i in range(k):
        for j in range(len(data)):
            tab[i][j] = y[data[:,i+1][j]]
    return mode(tab)[0][0]

print(ppv_k(iris.data,iris.target,2))
print("verification:")
print(ppv_k(iris.data,iris.target,2).all() == neigh_2.predict(iris.data).all())


# <h3 style="color:#8080C0">A. Classifieur Bayesien Naïf</h3>
# <h4>1. Créez une fonction CBN(X,Y) qui prend en entrée des données X et des étiquettes Y et qui renvoie une étiquette, pour chaque donnée, prédite à partir de la classe la plus probable selon l’équation (1). Ici encore, on prend chaque donnée, une par une, comme donnée de test et on considère toutes les données comme données d’apprentissage. Il est conseillé de calculer d’abord les barycentres et les probabilités à priori P(ωk) pour chaque classe, puis de calculer les probabilités conditionnelles P(xi/ωk) pour chaque classe et chaque variable.</h4>

# In[10]:


def cbn(x,y):
    ############ 
    # calculer le barycentre    
    dic = {}
    barycentre = np.zeros((len(np.unique(y)),x.shape[1]))

    for i in range(len(np.unique(y))):
        dic[i] = np.zeros((len(np.where(y==i)[0]),x.shape[1]))
        j=0
        for k in np.where(y==i)[0]:
            dic[i][j] = x[k]
            j=j+1
        barycentre[i] = np.mean(dic[i],axis = 0)
    ############
    # calculer le probabilite pwk
    pw = np.zeros(len(np.unique(y)))
    for i in range(len(np.unique(y))):
        pw[i] = sum(y == i)/y.size
    ###########
    #p conditionel p(xi|wk)
    p_cond = np.zeros((len(x),len(np.unique(y))))
    for k in range(len(np.unique(y))):
        for i in range(len(x)):
            sum_xb = 0
            dxk = 0
            dxk = metrics.pairwise.euclidean_distances([x[i],barycentre[k]])[0][1] #Dxk
            for j in range(len(np.unique(y))):
                sum_xb = sum_xb + metrics.pairwise.euclidean_distances([x[i],barycentre[j]])[0][1] #Dxk
            p_cond[i][k] = 1-(dxk/sum_xb)
    ##########
    #calculer la distribution du class
    class_x = np.zeros(len(x))
    p_final = np.zeros(len(np.unique(y)))
    for i in range(len(x)):
        for j in range(len(np.unique(y))):
                p_final[j] = p_cond[i][j]*pw[j]
        class_x[i] = np.argmax(p_final)
    
    return class_x
cbn(iris.data,iris.target)


# <h4>
# 2. La fonction CBN calcule une étiquette prédite pour chaque donnée. Modifiez la fonction pour calculer et renvoyer l’erreur de prédiction : c’est à dire le pourcentage d’étiquettes mal prédites. Testez sur les données Iris.</h4>

# In[11]:


def cbn_erreur(x,y):
    ############ 
    # calculer le barycentre    
    dic = {}
    barycentre = np.zeros((len(np.unique(y)),x.shape[1]))

    for i in range(len(np.unique(y))):
        dic[i] = np.zeros((len(np.where(y==i)[0]),x.shape[1]))
        j=0
        for k in np.where(y==i)[0]:
            dic[i][j] = x[k]
            j=j+1
        barycentre[i] = np.mean(dic[i],axis = 0)
    ############
    # calculer le probabilite pwk
    pw = np.zeros(len(np.unique(y)))
    for i in range(len(np.unique(y))):
        pw[i] = sum(y == i)/y.size
    ###########
    #p conditionel p(xi|wk)
    p_cond = np.zeros((len(x),len(np.unique(y))))
    for k in range(len(np.unique(y))):
        for i in range(len(x)):
            sum_xb = 0
            dxk = 0
            dxk = metrics.pairwise.euclidean_distances([x[i],barycentre[k]])[0][1] #Dxk
            for j in range(len(np.unique(y))):
                sum_xb = sum_xb + metrics.pairwise.euclidean_distances([x[i],barycentre[j]])[0][1] #Dxk
            p_cond[i][k] = 1-(dxk/sum_xb)
    ##########
    #calculer la distribution du class
    class_x = np.zeros(len(x))
    p_final = np.zeros(len(np.unique(y)))
    for i in range(len(x)):
        for j in range(len(np.unique(y))):
                p_final[j] = p_cond[i][j]*pw[j]
        class_x[i] = np.argmax(p_final)
    
    class_x = (class_x == y) 
    nb = np.sum(class_x == 0)
    return nb/len(y)
cbn_erreur(iris.data,iris.target)


# <h4>3. Testez la fonction du Classifieur Bayesien Naïf inclut dans sklearn. Cette fonc- tion utilise une distribution Gaussienne au lieu des distances aux barycentres. Les résultats sont-ils différents ?</h4>

# In[12]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(iris.data, iris.target)


# In[13]:


clf.predict(iris.data)


# In[14]:


np.sum((clf.predict(iris.data) == cbn(iris.data,iris.target)) == 0)/len(iris.target)


# In[15]:


np.sum((clf.predict(iris.data) == iris.target) == 0)/len(iris.target)


# <p style="color:green">Ils ont 6% de différence selon le résultat obtenu, mais l'erreur de prédiction de la méthode "GaussianNB" (0.04) est moins que la fonction CBN (0.07333333333333333)</p>

# In[16]:


'''fait par qn d'aure '''
def CBN(X, Y):
    # liste des moyennes (barycentre) de la classe
    moy = [np.mean(X[(Y==0)], axis=0), np.mean(X[(Y==1)], axis=0), np.mean(X[(Y==2)], axis=0)]
    # on convertit cette liste de moyenne en tableau numpy plus facile à manipuler
    moy=np.asarray(moy)

 

    # distance euclidienne entre une donnée X et le barycentre (moyenne) de la classe
    dxk = metrics.pairwise.euclidean_distances (X, moy)
    
    # somme des distances entre la donnée X et chaque barycentre de chaque classe
    dxb=np.sum(dxk, axis=1)
    pxk = np.zeros(450,dtype=float).reshape((150, 3))
    for i in range (0,len(dxb)):
        for j in range (0, 3):
            # probabilité qu'une donnée X ait la valeur Xi pour la variable i connaissant sa classe
            pxk[i][j]=1-(dxk[i][j]/dxb[i]) 
    
    # propabilité d'appartenance à chaque classe
    Ybay=np.argmax(pxk, axis=1)
    return Ybay


# In[17]:


CBN(iris.data,iris.target)


# In[ ]:




