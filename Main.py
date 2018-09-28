'''
Created on 24 de set de 2018
@author: gusta
'''

from projeto.streams.readers.arff_reader import ARFFReader
from sklearn.neighbors import NearestNeighbors
from projeto.Dynse import ClassificationEngine
from sklearn.naive_bayes import GaussianNB
from projeto.Dynse import PrunningEngine
from projeto.Dynse import Dynse
import pandas as pd
import numpy as np


def tratamentoDados(stream, vizinhos):
    '''
    metodo para ajustar o dataset conforme o artigo do Dynse
    '''
    
    # variavel para salvar o novo stream
    new_stream = []
    
    # ajustando o formato do stream
    stream = np.asarray(stream)
   
    # repetir ate o dataset ficar vazio
    while(len(stream)>vizinhos): 
        
        print("inicio:", len(stream))
    
        # dividindo os dados para treinamento e teste
        train = stream[:,0:-1]
        
        # ajustando o knn sobre os dados
        nbrs = NearestNeighbors(n_neighbors=vizinhos, algorithm='ball_tree')
        nbrs.fit(train)
        
        # sorteando uma isntancia aleatoria
        index = np.random.randint(low=0, high=len(stream))
        
        # obtendo a observacao sorteada
        obs = stream[index]
        
        # obtendo os vizinhos de obs
        _, indices = nbrs.kneighbors([obs[0:-1]])
        
        # removendo o indice pesquisado
        indices = indices[0]
        
        # salvando os vizinhos de obs no novo stream
        for j in indices:
            # salvando no novo dataset
            new_stream.append(stream[j])
            
        # excluindo os itens adicionados
        stream = np.delete(stream, indices, 0)
        
        # salvando as instancias de teste de forma aleatoria
        indices = []
        for x in range(vizinhos):
            
            # gerando um numero aleatorio
            if(x == 0):
                j = np.random.randint(low=0, high=len(stream))
                indices.append(j)
            else:
                while(True):
                    j = np.random.randint(low=0, high=len(stream))
                    if(j not in indices):
                        indices.append(j)
                        break
                    
            # salvando no novo dataset
            new_stream.append(stream[j])
            
        
        # excluindo os itens adicionados
        stream = np.delete(stream, indices, 0)
        
    
    # retornando o novo dataset
    return np.asarray(new_stream)

def main():
    
    # definindo o dataset
    #datasets = ['STAGGER', 'SEA', 'SEARec', 'optdigits', 'letter']
    #step_sizes = [500, 500, 220, 100, 100, 100]
    #train_sizes = [250, 250, 20, 50, 50, 50]
    
    datasets = ['optdigits', 'letter']
    step_sizes = [100, 100]
    train_sizes = [50, 50]
    
    for i in range(len(datasets)): 
    
        # definindo o mecanismo de classificacao
        engines = ['knorae', 'knorau', 'ola', 'lca', 'posteriori', 'priori']
        
        for j in range(len(engines)):

            # defininindo o mecanismo de poda        
            if(engines[j]=='knorae'):
                pruning = ['age', 'accuracy']
            else:
                pruning = ['age']
            
            # for para cada dataset
            for k in range(len(pruning)):
            
                #for para a quantidade de execucoes
                for x in range(1):
            
                    #1. importando o dataset
                    labels, _, stream_records = ARFFReader.read("projeto/data_streams/"+datasets[i]+".arff")

                    # tratamento dos dados com mudancas virtuais                    
                    if(datasets[i] == 'optdigits' or datasets[i] == 'letter'):
                        stream_records = tratamentoDados(stream_records, train_sizes[i])
                    
                    #2. instanciando o mecanismo de classificacao
                    ce = ClassificationEngine(engines[j])
                 
                    #3. definindo o criterio de poda
                    pe = PrunningEngine(pruning[k]) 
                       
                    #4. instanciando o classificador base
                    bc = GaussianNB()
                    
                    #5. instanciando o framework
                    dynse = Dynse(D=25,
                                  M=4, 
                                  K=5, 
                                  CE=ce, 
                                  PE=pe, 
                                  BC=bc)
                    
                    # para acompanhar a execucao
                    dynse.NAME = dynse.NAME+"-"+datasets[i]+"-"+str(x)
                     
                    #6. executando o framework
                    dynse.prequential(labels=labels, 
                                      stream=stream_records, 
                                      step_size=step_sizes[i],
                                      train_size=train_sizes[i])
                    
                    # salvando a predicao do sistema
                    df = pd.DataFrame(data={'target':dynse.TARGET, 'predictions': dynse.PREDICTIONS})
                    df.to_csv(dynse.NAME+".csv")
        
if __name__ == "__main__":
    main()        
