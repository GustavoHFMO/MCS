'''
Created on 24 de set de 2018
@author: gusta
'''

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# definindo o dataset
i = 2
datasets = ['STAGGER', 'SEA', 'SEARec']
step_sizes = [500, 500, 220]
train_sizes = [250, 250, 20] 

# definindo o mecanismo de classificacao
j = 3
engines = ['knorae', 'knorau', 'ola', 'lca', 'posteriori', 'priori']

# defininindo o mecanismo de poda
k = 0
pruning = ['age', 'accuracy']

acuracia_geral = []
for x in range(1):
    # lendo os arquivos e computando a acuracia
    string = "arquivos/DecisionTree-Dynse-"+engines[j]+"-"+pruning[k]+"-"+datasets[i]+"-"+str(x)+".csv"
    print(string)
    arquivo = pd.read_csv(string) 
    target = arquivo['target']
    prediction = arquivo['predictions']
    acuracia_geral.append(accuracy_score(target, prediction))

print(datasets[i]+ ": %.3f (%.3f)" % (np.mean(acuracia_geral), np.std(acuracia_geral)))
