#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;background-color:#336699;color:white;">TP6
# Analyse et prédiction des infections COVID-19</h1>
# <h3 style="color:#8080C0"><i style="color:red">L’objectif de ce TP</i> est d’explorer ces données pour en extraire des connaissances afin d’aider la communauté à mieux comprendre la propagation du COVID-19.
# <br/>Le TP est composé de l’ensemble des questions suivant :
# <br/>Afin d'analyser l'ensemble des données, vous devez identifier et extraire certaines informations statistiques sur les données, par exemple : le type de données, les valeurs manquantes, les valeurs aberrantes, la corrélation entre les variables, etc.
# <br>Dans le cas des valeurs manquantes, vous pouvez les remplacer par la moyenne, la médiane ou le mode de la variable concernée.</h3>

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: QIAN Xiaotong
@author: BIAN Yiping
"""
import pandas as pd
import numpy as np
import time

data = pd.read_csv("clean-hubei.csv")
data


# In[2]:


data.dtypes


# <h3 style = "color:Orange">Data pre-processing</h3>
# <h5 style = "color: #38B0DE">Etape 1 : D'abord on a vu pas mal de valeur manquantes, donc on va supprimer les colonnes qui contient qu'une seule type de valeur ou qui contient que "Nan". </h5>

# In[3]:


for index,row in data.iteritems():
    if len(data[index].unique()) == 1:
        print(index,' : ',data[index].unique())


# In[4]:


data = data.drop(['province','country','geo_resolution',
                  'travel_history_dates','travel_history_location',
                  'notes_for_discussion','location',
                  'admin1','country_new'],axis=1)


# <h5 style = "color: #38B0DE">Etape 2 : Ensuite on calcule le pourcentage de valeur manquante de chaque variable. Supprimer qui est plus grand que 0.999</h5>

# In[5]:


for index,row in data.iteritems():
       if (data[index].isna().sum()/len(data[index])>0.999):
               print(index,' : ',data[index].isna().sum()/len(data[index]))


# In[6]:


data = data.drop(['reported_market_exposure','sequence_available'],axis=1)


# <h5 style = "color: #38B0DE">Etape 3 : Ensuite on voit les variables qui contient les "date", et on choisi le mimum de la date de ces variables, ensuite on remplace les valeurs par la distance entre cette date et les valeurs.</h5>

# In[7]:


data['date_onset_symptoms'].unique()


# In[8]:


data['date_admission_hospital'].unique()


# In[9]:


data['date_confirmation'].unique()


# In[10]:


data['date_death_or_discharge'].unique()


# In[11]:


for i in range(data.shape[0]):
    for j in ['date_onset_symptoms','date_admission_hospital','date_confirmation','date_death_or_discharge']:
        if type(data.loc[i,j]) == str:
            if data.loc[i,j] == '- 18.01.2020':
                data.loc[i,j] = int(time.mktime(time.strptime(data.loc[i,j][2:], "%d.%m.%Y")))/86400 
            elif data.loc[i,j] == '01.01.2020 - 31.01.2020':
                data.loc[i,j] = (int(time.mktime(time.strptime(data.loc[i,j][13:24], "%d.%m.%Y"))) + int(time.mktime(time.strptime(data.loc[i,j][0:10], "%d.%m.%Y"))))/(2*86400) 
            else :
                data.loc[i,j] = int(time.mktime(time.strptime(data.loc[i,j], "%d.%m.%Y")))/86400   
            


# In[12]:


minimum = min([min(data['date_onset_symptoms']),
               min(data['date_admission_hospital']),
               min(data['date_confirmation']),
               min(data['date_death_or_discharge'])]) 
for j in ['date_onset_symptoms','date_admission_hospital','date_confirmation','date_death_or_discharge']:
        data.loc[np.where(data[j].isna() == False)[0],j] = data.loc[np.where(data[j].isna() == False)[0],j] - minimum


# In[13]:


for j in ['date_onset_symptoms','date_admission_hospital','date_confirmation','date_death_or_discharge']:
    data.loc[np.where(data[j].isna())[0],j] = data.loc[np.where(data[j].isna() == False)[0],j].mode()[0]
    data[j] = data[j].astype(int)


# In[14]:


data['date_onset_symptoms'].unique()


# In[15]:


data['date_admission_hospital'].unique()


# In[16]:


data['date_confirmation'].unique()


# In[17]:


data['date_death_or_discharge'].unique()


# <h5 style = "color: #38B0DE">Etape 4 : Ensuite on voit la variable "age" est de type object, plus précisement du type string, donc on faire transformer le type "str" à "float", et pour les valeurs qui est sous cette forme "a-b", on fait b-a, ensuite on remplace les valeurs manquantes par la moyenne.</h5>

# In[18]:


data['age'].unique()


# In[19]:


for i in range(data.shape[0]):
    if (type(data.loc[i,'age']) == str) :
        if data.loc[i,'age'].find('-') != -1:
            data.loc[i,'age'] = (float(data.loc[i,'age'].split('-')[1]) 
                                 + float(data.loc[i,'age'].split('-')[0]))/2.0
        else:
            data.loc[i,'age'] = float(data.loc[i,'age'])


# In[20]:


moyen = data.loc[np.where(data['age'].isna() == False)[0],'age'].mean()
data.loc[np.where(data['age'].isna())[0],'age'] = moyen
data['age'].unique()


# In[21]:


data['age'] = data['age'].astype(float)


# <h5 style = "color: #38B0DE">Etape 5 : Ensuite on voit il y a trois type de valeurs dans la variable "outcome", donc en respectant le consignes du professeur, on remplace la valeur manquante par "discharged".</h5>

# In[22]:


data['outcome'].unique()


# In[23]:


data.loc[np.where(data['outcome'].isna())[0],'outcome'] = 'discharged'
data['outcome'].unique()


# <h5 style = "color: #38B0DE">Etape 6 : Ensuite on voit il y a pas mal de valeur de type "string", pour faciliter le calcul de la coefficient de la corrélation, on les remplacer par la valeur numérique.</h5>

# In[24]:


dic_correspont={} # Stocker les indice correspondants apres factorize
for index,row in data.iteritems():
    if (index != 'age') & (data[index].dtype != int):
        dic_correspont[index] = pd.factorize(data[index]) 
        data[index] = dic_correspont[index][0]
dic_correspont


# <h5 style = "color: #38B0DE">Etape 7 : Après on remplace tous les valeurs manquantes par le mode.</h5>

# In[25]:


for index,row in data.iteritems():
    if (data[index].isin([-1]).any() == True):
        data.loc[np.where(data[index] == -1)[0],index] = data.loc[np.where(data[index] != -1)[0],index].mode()[0]


# In[26]:


data


# <h4>1. Calculez les corrélations entre les variables. Quelles sont variables les plus corrélées avec la cible (‘result’)? Expliquez les résultats.</h4>

# In[27]:


correlation = data.corr(method='pearson')


# In[28]:


correlation


# In[29]:


correlation['outcome']


# <p style="color: green">Selon le résultat obtenu, on peut voir clairment la variable "<strong style="color:orange">symptoms</strong>", "<strong style="color:orange">date_admission_hospital</strong>","<strong style="color:orange">age</strong>","<strong style="color:orange">additional_information </strong>" les plus corrélées avec la cible ("<strong style="color:orange">outcome</strong>"). Le résultat n'est pas très surpris, car  évidement la date admission à l'hôpital a une forte lien avec le résultat, et aussi si la personne a une maladie chrononique(qui est décrit dans la colonne "additional_information"), elle va avoir une grande possibilité d'être mort après avoir contaminée par Covid-19, et aussi on a vu qu'il y a plus de personnes agées qui est mort à cause du Covid-19 dans les nouvelles donc, on ne peut pas oublier la corrélation entre age et le résultat</p>

# <h4>2. Visualisez les données en deux dimensions en passant par l’ACP (analyse en composantes principales). Pouvez-vous utiliser une autre méthode ? </h4>

# In[30]:


from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

label = data['outcome']

pca = PCA(n_components=2)
x_p = pca.fit(data).transform(data)
print(x_p)
x = x_p[:,0]
y = x_p[:,1]

plt.figure()
plt.title('apres la methode pca')
plt.scatter(x, y,c=label)
plt.xlabel('dimension 1')
plt.ylabel('diemnsion 2')


# <p style="color:green">Donc la couleur jaune représente les personnes mort </p>
# <i style="color:blue">On peut aussi utiliser la méthode NMF</i>

# In[31]:


from sklearn.decomposition import NMF

nmf = NMF(n_components=2)
x_n = nmf.fit(data).transform(data)
print(x_n)
x = x_n[:,0]
y = x_n[:,1]

plt.figure()
plt.title('apres la methode NMF')
plt.scatter(x, y,c=label)
plt.xlabel('dimension 1')
plt.ylabel('diemnsion 2')


# <h3 style="color:#8080C0">
# Dans la suite, nous utilisons une méthode d'apprentissage automatique afin de prédire la classe : les patients sont soit «décédés» (‘died’) soit «sortis» (‘discharged’) de l'hôpital. Vous pouvez utiliser la classification par K-Nearest Neighbours (K-NN), l’arbre de decision ou le classificateur Bayes.</h3>

# In[42]:


from sklearn.model_selection import train_test_split

X_train_outcome, X_test_outcome, y_train_outcome, y_test_outcome = train_test_split(PCA(n_components=2).fit(data.drop(['outcome'],axis=1)).transform(data.drop(['outcome'],axis=1)), data['outcome'], test_size=0.2, random_state=None)


# In[43]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(X_train_outcome,  y_train_outcome)
len(clf.predict(X_test_outcome))


# <h4>3. Les résultats obtenus doivent être validés en utilisant certains indices externes comme l’erreur de prédiction (matrice de confusion et précision) ou d'autres comme Rappel, F-Measure, ...</h4>

# In[44]:


from sklearn.metrics import confusion_matrix
y_true = y_test_outcome
y_pred = clf.predict(X_test_outcome)
conf_mat = confusion_matrix(y_true, y_pred)
conf_mat


# In[46]:


#accuracy = (TP + TN) / (TP + FP + TN + FN)
accuracy = (conf_mat[0][0] + conf_mat[1][1]) / conf_mat.sum()
accuracy


# <h4>4. Utilisez la régression pour prédire l'âge (age) des personnes en fonction d'autres variables. Vous avez le choix sur ces variables explicatives ? Comment choisissez-vous ces variables ? Calculez la qualité de la prédiction à l'aide de l'erreur MSE (Mean Squared Error).</h4>

# In[47]:


correlation['age']


# <p style="color:green">Voir les coefficients de la corrélation entre "age" et les autres variables. Selon le résultat, j'ai décidé de prendre les variables "outcome",'symptoms' et 'additional_information' pour faire la prédiction.</p>

# In[50]:


X_train, X_test, y_train, y_test = train_test_split(data[['outcome','symptoms','additional_information']], data['age'], test_size=0.2, random_state=None)


# In[51]:


from sklearn.linear_model import LinearRegression

lrModel = LinearRegression()
lrModel.fit(X_train,y_train)

lrModel.predict(X_test)


# In[52]:


from sklearn.metrics import mean_squared_error
y_true = y_test
y_pred = lrModel.predict(X_test)
mean_squared_error(y_true, y_pred)


# <h4>
# 5. Appliquer trois méthodes de clustering (K-means, NMF et CAH) sur l'ensemble de données pour segmenter les personnes en différents groupes. Utilisez l'index de Silhouette pour connaître le meilleur nombre de clusters.</h4>

# In[53]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_kmeans = np.zeros(10) 
for i in range(10):
    kmeans_model = KMeans(n_clusters=i+2).fit(x_p)
    silhouette_kmeans[i] = silhouette_score(x_p, kmeans_model.labels_)
plt.figure()
plt.plot(range(1,11),silhouette_kmeans)


# In[129]:


# pas encore trouver la méthode pour calculer silhouette_score du NMF


# In[54]:


from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(x_p, 'ward')
plt.figure(figsize=(50, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.show()


# In[55]:


from scipy.cluster.hierarchy import fcluster

silhouette_cah = np.zeros(10) 
for i in range(10):
    Z = linkage(x_p, 'ward')
    silhouette_cah[i] = silhouette_score(x_p, fcluster(Z, i+2, criterion='maxclust'))
plt.figure()
plt.plot(range(2,12),silhouette_cah)


# <h4>6. Visualisez les résultats à l'aide de scatter pour analyser visuellement la structure de clustering des trois méthodes.</h4>

# In[138]:


x = x_p[:,0]
y = x_p[:,1]

label = KMeans(n_clusters=3).fit(x_p).labels_

plt.figure(figsize=(15,5))
plt.title('apres la methode KMeans')
plt.scatter(x, y, c=label)
plt.xlabel('dimension 1')
plt.ylabel('diemnsion 2')


# In[56]:


label = fcluster(Z, 4, criterion='maxclust')

plt.figure(figsize=(15,5))
plt.title('apres la methode CAH')
plt.scatter(x, y, c=label)
plt.xlabel('dimension 1')
plt.ylabel('diemnsion 2')


# <h4>7. Les données sont déséquilibrées. Vous pouvez les équilibrer en réduisant aléatoirement la classe majoritaire. Supposons que vous extrayez aléatoirement des échantillons équilibrés. Comment les résultats de la prédiction changeront-ils?</h4>

# In[109]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(data[data['sex'] == 1], data[data['sex'] == 1]['outcome'], test_size=0.1, random_state=None)
x_train0, x_test0, y_train0, y_test0 = train_test_split(data[data['sex'] == 0], data[data['sex'] == 0]['outcome'], test_size=0.996, random_state=None)


# In[110]:


len(x_train1)


# In[111]:


len(x_train0)


# In[113]:


x_train = pd.concat([x_train0,x_train1],axis = 0)
x_test = pd.concat([x_test0,x_test1],axis = 0)
y_train = pd.concat([y_train0,y_train1],axis = 0)
y_test = pd.concat([y_test0,y_test1],axis = 0)

clf = GaussianNB().fit(x_train,  y_train)
y_true = y_test
y_pred = clf.predict(x_test)
conf_mat = confusion_matrix(y_true, y_pred)
conf_mat


# In[115]:


#accuracy = (TP + TN) / (TP + FP + TN + FN)
accuracy = (conf_mat[0][0] + conf_mat[1][1]) / conf_mat.sum()
accuracy


# In[118]:


max(data['age'])


# In[119]:


min(data['age'])


# In[127]:


len(data[(data['age'] >= 30) & (data['age'] < 45)])


# In[128]:


len(data[(data['age'] >= 45) & (data['age'] < 60)])


# In[129]:


len(data[(data['age'] >= 60) & (data['age'] < 75)])


# In[130]:


len(data[(data['age'] >= 75) & (data['age'] < 90)])


# In[131]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(data[(data['age'] >= 30) & (data['age'] < 45)], data[(data['age'] >= 30) & (data['age'] < 45)]['outcome'], test_size=0.1, random_state=None)
x_train2, x_test2, y_train2, y_test2 = train_test_split(data[(data['age'] >= 45) & (data['age'] < 60)], data[(data['age'] >= 45) & (data['age'] < 60)]['outcome'], test_size=0.996, random_state=None)
x_train3, x_test3, y_train3, y_test3 = train_test_split(data[(data['age'] >= 60) & (data['age'] < 75)], data[(data['age'] >= 60) & (data['age'] < 75)]['outcome'], test_size=0.1, random_state=None)
x_train4, x_test4, y_train4, y_test4 = train_test_split(data[(data['age'] >= 75) & (data['age'] < 90)], data[(data['age'] >= 75) & (data['age'] < 90)]['outcome'], test_size=0.1, random_state=None)


# In[132]:


len(x_train1)


# In[133]:


len(x_train2)


# In[134]:


len(x_train3)


# In[135]:


len(x_train4)


# In[136]:


x_train = pd.concat([x_train1,x_train2,x_train3,x_train4],axis = 0)
x_test = pd.concat([x_test1,x_test2,x_test3,x_test4],axis = 0)
y_train = pd.concat([y_train1,y_train2,y_train3,y_train4],axis = 0)
y_test = pd.concat([y_test1,y_test2,y_test3,y_test4],axis = 0)

clf = GaussianNB().fit(x_train,  y_train)
y_true = y_test
y_pred = clf.predict(x_test)
conf_mat = confusion_matrix(y_true, y_pred)
# TP FN
# FP TN
conf_mat


# In[138]:


#accuracy = (TP + TN) / (TP + FP + TN + FN)
accuracy = (conf_mat[0][0] + conf_mat[1][1]) / conf_mat.sum()
accuracy


# In[139]:


label = x_train['outcome']

pca = PCA(n_components=2)
x_p = pca.fit(x_train).transform(x_train)
x = x_p[:,0]
y = x_p[:,1]

plt.figure()
plt.title('apres la methode pca')
plt.scatter(x, y,c=label)
plt.xlabel('dimension 1')
plt.ylabel('diemnsion 2')


# In[140]:


label = KMeans(n_clusters=3).fit(x_p).labels_

plt.figure(figsize=(15,5))
plt.title('apres la methode KMeans')
plt.scatter(x, y, c=label)
plt.xlabel('dimension 1')
plt.ylabel('diemnsion 2')


# <p style="color:green">Selon les deux résultat obtenu, on a remarqué la qualité de la prédiction augmente </p>

# <h4>8. Comment pouvez-vous mieux gérer ce déséquilibre entre les classes ?</h4>
# <p style="color:green">On a remarqué qu'il y a pas mal des valeurs manquantes dans cette base de données, et on les remplacer par la moyenne ou par la mode, cela va amener des désquilibres entre les classes, et on essaie de trouver les variables qui ont forte lien avec la variable concerné pour la prédiction, en voyant le coefficient de la corrélation de chaque variable, et essaie de prendre le même nombre des valeurs aléatoirement de chaque type de valeur de la variable. Par exemple, Voir en dessous, on sais qu'il y a des déséquilibrement entre l'homme et femme dans la base de données d'entraînement, donc on essaie de prendre aléatoirement le nombre de valeurs quasiment le même entre l'homme et femme, ce qui aussi avoir des risque de d'autre part, et ne peut pas tout simplement dire que le modèle est le meuilleur, car on a finalement 75 données d'entraînement, pas 10135 de la base de données originale. D'ailleurs on peut aussi considérer la variable "age" qui ont aussi un très important lien avec le résultat. </p>

# <h4>9. Pour trouver les meilleurs paramètres pour les modèles, l'algorithme Greedy Search peut être utilisé, disponible dans la bibliothèque scikit-learn. Expliquez l'algorithme et utilisez-le pour les modèles d'apprentissage choisis afin de trouver les meilleurs paramètres.</h4>

# <h4>10. Présentez et expliquez le formalisme algorithmique et mathématique de la méthode qui donne les meilleurs résultats. Expliquez tous les paramètres de la méthode utilisée et leur impact sur les résultats.</h4>
