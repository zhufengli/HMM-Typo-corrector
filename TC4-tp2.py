#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Ce TP continue le TP précédent. Nous allons reprendre d'ailleurs les mêmes données et commencer la mise en oeuvre d'un modèle de Markov pour la prédiction des étiquettes: 
# * une observation est une phrase, représentée comme une séquence de variables aléatoires, une par mot de la phrase
# * à cette observation est associée une séquence de variables aléatoires représentant les étiquettes, une par mot de la phrase également
# 
# On suppose que la séquence d'observation (une phrase) est générée par un modèle de Markov caché. Les variables cachées sont donc les étiquettes à inférer. Nous allons commencer par écrire une classe python pour représenter le HMM. Cette classe évoluera au fil des TPs. 
# 
# Pour cela le code de départ suivant est donné. Afin d'initialiser un HMM, nous devons connaitre : 
# - l'ensemble des états (ou *state_list*), dans notre cas l'ensemble des étiquettes grammaticales;
# - l'ensemble des observations (ou *observation_list*), dans notre cas l'ensemble des mots connus; tous les autres mots seront remplacés par l'élément spécial *UNK* qui fait partie de l'ensemble des observations. 
# 
# Enfin, en interne il est plus facile d'indexer les mots et et les états par des entiers. Ainsi à chaque éléments de respectivement l'ensemble des états et l'ensemble des observations, est associé un indice. Cela nous permet de tout traiter en "matricielle". 

# In[2]:


import nltk
from numpy import array, ones, zeros
import sys

# Some words in test could be unseen during training, or out of the vocabulary (OOV) even during the training. 
# To manage OOVs, all words out the vocabulary are mapped on a special token: UNK defined as follows: 
UNK = "<unk>" 
UNKid = 0 

class HMM:
        def __init__(self, state_list, observation_list,
                 transition_proba = None,
                 observation_proba = None,
                 initial_state_proba = None):
            """Builds a new Hidden Markov Model
            state_list is the list of state symbols [q_0...q_(N-1)]
            observation_list is the list of observation symbols [v_0...v_(M-1)]
            transition_proba is the transition probability matrix
                [a_ij] a_ij = Pr(Y_(t+1)=q_i|Y_t=q_j)
            observation_proba is the observation probablility matrix
                [b_ki] b_ki = Pr(X_t=v_k|Y_t=q_i)
            initial_state_proba is the initial state distribution
                [pi_i] pi_i = Pr(Y_0=q_i)"""
            print ("HMM creating with: ")
            self.N = len(state_list) # The number of states
            self.M = len(observation_list) # The number of words in the vocabulary
            print (str(self.N)+" states")
            print (str(self.M)+" observations")
            self.omega_Y = state_list # Keep the vocabulary of tags
            self.omega_X = observation_list # Keep the vocabulary of tags
            # Init. of the 3 distributions : observation, transition and initial states
            if transition_proba is None:
                self.transition_proba = zeros( (self.N, self.N), float) 
            else:
                self.transition_proba=transition_proba
            if observation_proba is None:
                self.observation_proba = zeros( (self.M, self.N), float) 
            else:
                self.observation_proba=observation_proba
            if initial_state_proba is None:
                self.initial_state_proba = zeros( (self.N,), float ) 
            else:
                self.initial_state_proba=initial_state_proba
            # Since everything will be stored in numpy arrays, it is more convenient and compact to 
            # handle words and tags as indices (integer) for a direct access. However, we also need 
            # to keep the mapping between strings (word or tag) and indices. 
            self.make_indexes()

        def make_indexes(self):
            """Creates the reverse table that maps states/observations names
            to their index in the probabilities arrays"""
            self.Y_index = {}
            for i in range(self.N):
                self.Y_index[self.omega_Y[i]] = i
            self.X_index = {}
            for i in range(self.M):
                self.X_index[self.omega_X[i]] = i
      


# # Interface avec les données et apprentissage supervisé
# 
# Ainsi pour initialiser un HMM, nous allons devoir lire les données (chose faîte lors du TP précédent): 
# * écrire une fonction permettant d'initialiser le HMM à partir des données d'apprentissage
# * écrire une fonction *apprentissage_supervisé* qui permet d'estimer les paramètres 
# 
# Dans un premier temps, nous limiterons, comme lors du TP précédent, le vocabulaire aux mots apparaissant 10 fois ou plus. Les autres mots sont tous remplacés par la même forme *unk*
# 
# Pour cela, le plan de travail peut être envisagé ainsi: 
# * Lire les données puis générer un corpus de **train** (80%) puis de **test** (10%)
# * écrire une fonction qui créer à partir des données d'apprentissage (**train**), tous les comptes nécessaires pour l'estimation supervisée des paramètres du HMM
# * écrire 3 fonctions qui estimes les paramètres à partir des comptes, une fonction par distribution: observation, transition, état initial. 
# * écrire une fonction qui reprend le tout et qui estime tous les paramètres du HMM
# 
# 
# # Exercice : Algorithme de Viterbi
# 
# La question qui se pose est comment calculer la meilleure séquence d'étiquettes pour une phrase donnée connaissant les paramètres du HMM. Par meilleure, on entend la séquence d'étiquettes (ou d'états) la plus probable connaissant la séquence d'obervation. 
# 
# Proposer et implémenter un algorithme répondant à cette question. Pour vous aider à démarrer, cet algorithme s'appelle Viterbi et regardez cette vidéo https://www.youtube.com/watch?v=RwwfUICZLsA, pour comprendre comment il opère. 
# 
# # TODO pour la prochaine fois
# 
# * Finir la partie interface (qui comprend l'apprentissage supervisé)
# * Regarder la vidéo et implémenter l'algorithme de Viterbi
# 
# 
# 

# In[3]:


#lire des données
import numpy as np
import pickle
data = pickle.load(open( "brown.save.p", "rb" ))

#créer un dictionaire pour les étiquettes, dans la suite on va utiliser les listes au lieu de dictionaire

#state_list
etiquette={'0':0,'DET':1, 'NOUN':2, 'ADJ':3, 'VERB':4, 'ADP':5, '.':6, 'ADV':7, 'CONJ':8, 'PRT':9, 'PRON':10, 'NUM':11, 'X':12}

#Générer training set et test set
def create_set(data, train_per, dev_per):
    #separer des données suivant leur pourcentage
    train_set=[]
    dev_set=[]
    test_set=[]
    num=int(train_per*len(data))
    train_set=data[0:num]
    num2=int(dev_per*len(data))
    dev_set=data[num+1:(num+num2)]
    test_set=data[(num+num2)+1:len(data)]
    return train_set, dev_set, test_set

def map_function(data, etiquette):
    #a map function
    association={}
    for phrase in range(len(data)):
        for mot in range(len(data[phrase])):
            if data[phrase][mot][0] not in association.keys():
                association[data[phrase][mot][0]] = np.zeros(13)
            association[data[phrase][mot][0]][etiquette[data[phrase][mot][1]]]+=1
            
    association["<unk>"]=np.zeros(13)
    unk_list=[]
    
    #observation_list
    dictionary=[]
    
    for mot in association.keys():
        occurence=association[mot].sum()
        if occurence<10:
            unk_list.append(mot)
            association["<unk>"][0]+=occurence
        else:
            dictionary.append(mot)
            
    for mot in unk_list:
        del association[mot]
        
    return association, dictionary





'''
def préprocesse(data)
    for phrase in range(len(data)):
        for mot in range(len(data[phrase])):
            if data[phrase][mot][0] not in dictionary:
                data[phrase][mot][0]="<unk>"
                data[phrase][mot][1]="0"
    return data
'''


# In[4]:


train_set, dev_set, test_set=create_set(data, 80, 10)

map_train, dictionary = map_function(train_set, etiquette)

print(map_train["<unk>"])

print (dictionary.index("<unk>"))


# In[12]:


#Dans la suite il suffit de calculer les paramètres transition_proba, observation_proba, initial_state_proba

def calculate_transition_proba(data, etiquette):
    transition_proba = np.zeros( (len(etiquette), len(etiquette)), float) 
    #compter chaque transition, chaque colonne diviser par la somme colonne.
    one_transition=[None, None] #transition[0] état d'arriver, transition[1] état de sortie
    for phrase in range(len(data)):
        for mot in range(len(data[phrase])):
            if mot<len(data[phrase])-1:
                etat_arrive=etiquette[data[phrase][mot+1][1]] 
                etat_depart=etiquette[data[phrase][mot][1]]
                transition_proba[etat_arrive,etat_depart]+=1
    for i in range(len(etiquette)):
        transition_proba[:][i]=transition_proba[:][i]/transition_proba[:][i].sum()
    return transition_proba

def calculate_observation_proba(data, dictionary, etiquette):
    observation_proba = np.zeros( (len(dictionary), len(etiquette)), float)
    #compter P(X|Y), chaque colonne diviser par la somme colonne, presque comme dans TP1
    return observation_proba

def calculate_initial_state(data, etiquette):
    initial_state_proba = np.zeros(len(etiquette), float ) 
    #juste compter l'état du premier mot, et divise par la somme
    return initial_state_proba


# In[13]:


transition_proba = calculate_transition_proba(train_set, etiquette)
print(transition_proba)


# In[14]:


print (train_set)


# In[ ]:




