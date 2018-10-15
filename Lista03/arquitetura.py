'''
Created on 13 de out de 2018
@author: gusta
'''

from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.cross_validation import StratifiedKFold
from deslib.dcs.ola import OLA
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Arquitetura:
    def __init__(self, n_vizinhos):
        '''
        :n_vizinhos: quantidade de vizinhos mais proximos que serao utilizados para regiao de competencia
        '''
        
        self.n_vizinhos = n_vizinhos
    
    def kDN(self, x, y):
        '''
        Metodo para computar o grau de dificuldade de cada observacao em um conjunto de dados
        :param: x: padroes dos dados
        :param: y: respectivos rotulos
        :return: dificuldades: vetor com a probabilidade de cada instancia 
        '''
        
        # instanciando os vizinhos mais proximos
        nbrs = NearestNeighbors(n_neighbors=self.n_vizinhos+1, algorithm='ball_tree').fit(x)
        
        # variavel para salvar as probabilidades
        dificuldades = []
        
        # for para cada instancia do dataset
        for i in range(len(x)):
            
            # computando os vizinhos mais proximos para cada instancia
            _, indices = nbrs.kneighbors([x[i]])
            
            # verificando o rotulo dos vizinhos
            cont = 0
            for j in indices[0]:
                if(j != i and y[j] != y[i]):
                    cont += 1
                    
            # computando a porcentagem
            dificuldades.append(cont/(self.n_vizinhos+1))
    
        
        return dificuldades

    def neighbors(self, dsel, x_query):
        '''
        metodo para retornar apenas os indices dos vizinhos
        '''
        
        # instanciando os vizinhos mais proximos
        nbrs = NearestNeighbors(n_neighbors=self.n_vizinhos+1, algorithm='ball_tree').fit(dsel)
        
        # computando os vizinhos mais proximos para cada instancia
        _, indices = nbrs.kneighbors([x_query])
        
        return indices
        
    def fit(self, x, y, dsel_x, dsel_y):
        '''
        metodo para treinar a arquitetura de dois niveis
        :x: dados para treinamento
        :y: rotulo dos dados
        :dsel_x: padroes da janela de validacao
        :dsel_y: rotulos da janela de validacao
        '''
        
        # salvando as dificuldades das instancias
        self.H = self.kDN(x, y)
        
        # treinando o nivel 1 #########################################
        self.levelone = SVC(self.n_vizinhos)
        self.levelone.fit(x, y)
        
        # realizando a previsao para o conjunto de treinamento
        y_pred = self.levelone.predict(x)
        
        # salvando os indices das instancias que foram classificadas erradas
        indices = [i for i in range(len(y_pred)) if y_pred[i] != y[i]]
                
        # obtendo as dificuldades
        dificuldades = [self.H[i] for i in indices]
        
        # retirando a media do kdn dessas instancias
        self.limiar = np.min(dificuldades)
        ###############################################################
        
        # treinando o nivel 2 #########################################
        # salvando a jaela de validacao
        self.dsel_x = x
        self.dsel_y = y
        
        # criando o ensemble
        self.ensemble = BaggingClassifier(base_estimator=Perceptron(), 
                                            max_samples=0.9, 
                                            max_features=1.0, 
                                            n_estimators=100)
        self.ensemble.fit(x, y)
        
        # treinando o modelo 2
        self.leveltwo = OLA(self.ensemble.estimators_, self.n_vizinhos)
        self.leveltwo.fit(self.dsel_x, self.dsel_y)
        ###############################################################
    
    def predict_one(self, x):
        '''
        metodo para computar a previsao de um exemplo
        :x: padrao a ser predito
        '''
        
        # obtendo a vizinhanca do exemplo
        indices = self.neighbors(self.dsel_x, x)[0]
        
        # dificuldade da regiao
        dificuldades = [self.H[i] for i in indices]
        
        # media da dificuldadde da regiao
        media = np.max(dificuldades)
        
        # verificando a dificuldade da instancia
        if(media >= self.limiar):
            return self.leveltwo.predict(np.array([x]))[0]
        else:
            return self.levelone.predict(np.array([x]))[0]
    
    def predict(self, x):
        '''
        metodo para computar a previsao de um exemplo
        :x: padrao a ser predito
        '''
        
        # to predict multiple examples
        if(len(x.shape) > 1):
            # returning all labels
            return [self.predict_one(pattern) for pattern in x]
            
        # to predict only one example
        else:
            return self.predict_one(x)
      
def main():
    
    
    # importando o dataset
    data = pd.read_csv('dataset/kc2.csv')
        
    # obtendo os padroes e seus respectivos rotulos
    df_x = np.asarray(data.iloc[:,0:-1])
    df_y = np.asarray(data.iloc[:,-1])
    
    # dividindo os dados em folds
    skf = StratifiedKFold(df_y, n_folds=10)
                
    # tomando os indices para treinamento e teste
    train_index, test_index = next(iter(skf))
                        
    # obtendo os conjuntos de dados para treinamento e teste
    x_train = df_x[train_index]
    y_train = df_y[train_index]
    x_test = df_x[test_index]
    y_test = df_y[test_index]

    # importando o metodo
    arq = Arquitetura(7)
    
    # treinando o metodo
    arq.fit(x_train, y_train, x_train, y_train)
    
    # making predictions
    predictions = arq.predict(x_train)
    # evaluating the accuracy for the training dataset
    train_accuracy = np.mean(predictions == y_train) * 100
    print('train accuracy: %.1f' % train_accuracy)
    
    # making predictions
    predictions = arq.predict(x_test)
    # evaluating the accuracy for the test dataset
    test_accuracy = np.mean(predictions == y_test) * 100
    print('test accuracy: %.1f' % test_accuracy)
    
if __name__ == '__main__':
    main()