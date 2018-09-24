'''
Created on 24 de set de 2018
@author: gusta
'''

from projeto.streams.readers.arff_reader import ARFFReader
from projeto.Dynse import ClassificationEngine
from sklearn.naive_bayes import GaussianNB
from projeto.Dynse import PrunningEngine
from projeto.Dynse import Dynse
import pandas as pd

def main():
    
    # definindo o dataset
    #i = 3
    datasets = ['SEA', 'SEARec', 'STAGGER', 'For']
    step_sizes = [500, 500, 220, 100, 2200]
    train_sizes = [250, 250, 20, 50, 200] 
    
    # definindo o mecanismo de classificacao
    j = 0
    engines = ['knorae', 'knorau', 'ola', 'lca', 'posteriori', 'priori']
    
    # defininindo o mecanismo de poda
    k = 0
    pruning = ['age', 'accuracy']
    
    # for para cada dataset
    for i in range(len(datasets)):
    
        #for para a quantidade de execucoes
        for x in range(10):
    
            #1. importando o dataset
            labels, _, stream_records = ARFFReader.read("data_streams/"+datasets[i]+".arff")
            
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
             
            #6. executando o framework
            dynse.prequential(labels=labels, 
                              stream=stream_records, 
                              step_size=step_sizes[i],
                              train_size=train_sizes[i])
            
            # printando a acuracia final do sistema
            print(dynse.accuracyGeneral())
            
            # salvando a predicao do sistema
            df = pd.DataFrame(data={'target':dynse.TARGET, 'predictions': dynse.PREDICTIONS})
            df.to_csv(dynse.NAME+"-"+datasets[i]+"-"+x+".csv")
        
if __name__ == "__main__":
    main()        
