'''
Created on 24 de set de 2018
@author: gusta
'''

from Projeto.streams.readers.arff_reader import ARFFReader
from sklearn.cross_validation import StratifiedKFold
from Projeto.Dynse import ClassificationEngine
from sklearn.naive_bayes import GaussianNB
from Projeto.Dynse import PrunningEngine
from Projeto.Dynse import Dynse
import pandas as pd
import numpy as np


def tratamentoDados(stream, train_size):
    '''
    metodo para ajustar o dataset conforme o artigo do Dynse
    '''
    
    # obtendo os padroes e seus respectivos rotulos
    stream = np.asarray(stream)
    df_x = stream[:,0:-1]
    df_y = stream[:,-1]
    
    # quantidade de folds
    qtd_folds = 10
    
    # dividindo os dados em folds
    kf = KFold(n_splits=qtd_folds, random_state=None, shuffle=False)

    # tomando os indices para treinamento e teste
    for train_index, test_index in kf.split(df_x):
        x_train, x_test = df_x[train_index], df_x[test_index]
        y_train, y_test = df_y[train_index], df_y[test_index]

    # juntando os dados
    stream_train = np.zeros((x_train.shape[0], x_train.shape[1]+1), dtype='str')
    stream_train[:, 0:-1] = x_train
    stream_train[:,-1] = y_train
    
    stream_test = np.zeros((x_test.shape[0], x_test.shape[1]+1), dtype='str')
    stream_test[:, 0:-1] = x_test
    stream_test[:,-1] = y_test
    
    # transformando em stream
    stream = np.concatenate((stream_train, stream_test))

    # retornando o novo dataset
    return stream

def main():
    
    # definindo o dataset
    #datasets = ['STAGGER', 'SEA', 'SEARec', 'optdigits', 'letter']
    #step_sizes = [500, 500, 220, 100, 100, 100]
    #train_sizes = [250, 250, 20, 50, 50, 50]
    
    datasets = ['optdigits']
    step_sizes = [100]
    train_sizes = [50]
    
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
                    labels, _, stream_records = ARFFReader.read("Projeto/data_streams/"+datasets[i]+".arff")

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
                    dynse.prequential_batch(labels=labels, 
                                      stream=stream_records, 
                                      step_size=step_sizes[i],
                                      train_size=train_sizes[i])
                    
                    # salvando a predicao do sistema
                    df = pd.DataFrame(data={'target':dynse.TARGET, 'predictions': dynse.PREDICTIONS})
                    df.to_csv(dynse.NAME+".csv")
        
if __name__ == "__main__":
    main()        
