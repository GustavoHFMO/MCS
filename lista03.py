'''
Created on 13 de out de 2018
@author: gusta
'''


from Lista03.geradores_tabela.Tabela_excel import Tabela_excel
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.cross_validation import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from Lista03.arquitetura import Arquitetura
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.dcs.lca import LCA
from deslib.dcs.ola import OLA 
from sklearn import metrics
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def printar_resultados(y_test, pred, nome_modelo):
    '''
    metodo para printar os resultados de cada modelo
    :param: y_test: dados correspondentes a saida de teste
    :param: pred: dados correspondentes a previsao do modelo
    :return: retorna as metricas: acuracia, auc, f1score e gmean
    '''
    
    # computando as metricas para os dados recebidos
    acuracia = metrics.accuracy_score(y_test, pred)
    auc = metrics.roc_auc_score(y_test, pred)
    f1measure = metrics.f1_score(y_test, pred, average='binary')
    gmean = geometric_mean_score(y_test, pred, average='binary')
    
    # calculando o desempenho
    print('\n'+nome_modelo)
    print("taxa de acerto:", acuracia)
    print("AUC:", auc)
    print("f-measure:", f1measure)
    print("g-mean:", gmean)
    
    # retornando as metricas
    return  acuracia, auc, f1measure, gmean

def escolher_modelo(nome, x_sel, y_sel, P, k):
    '''
    metodo para chamar o tipo de DS
    :param: x_sel: dados de treinamento da janela de validacao
    :param: y_sel: rotulos da janela de validacao
    :param: P: pool de classificadores
    :param: k: vizinhanca
    '''
        
    # escolhendo a tecnica de selecao de classificadores
    if(nome=='OLA'):
        DS = OLA(P, k)
        number_model = 0
        
    elif(nome=='LCA'):
        DS = LCA(P, k)
        number_model = 1
        
    elif(nome=='KNORAE'):
        DS = KNORAE(P, k)
        number_model = 2
        
    elif(nome=='KNORAU'):
        DS = KNORAU(P, k)
        number_model = 3
            
    # encontrando os classificadores competentes do DS escolhido
    DS.fit(x_sel, y_sel)
    
    # retornando a tecnica de DS
    return DS, number_model
    
def executar_modelo(nome, x_train, y_train, x_test, y_test, estimators, n_vizinhos, nome_datasets, h, j, tabela):
    '''
    metodo para executar os modelos
    '''
    
    # selecionando a tecnica de DS
    ds, num_model = escolher_modelo(nome, x_train, y_train, estimators, n_vizinhos)
    
    # computando a previsao
    pred = ds.predict(x_test)
                
    # printando os resultados
    acuracia, auc, f1measure, gmean = printar_resultados(y_test, pred, nome_datasets[h]+'-'+nome+'-['+str(j)+']')
            
    # escrevendo os resultados obtidos
    tabela.Adicionar_Sheet_Linha(num_model, j, [acuracia, auc, f1measure, gmean])

def main():
    
    # 1. Definindo variaveis para o experimento #########################################################################
    qtd_modelos = 100
    qtd_execucoes = 30
    qtd_amostras = 0.9
    qtd_folds = 10
    n_vizinhos = 7
    nome_datasets = ['kc1', 'kc2']
    # 1. End ############################################################################################################

    # for para variar entre os datasets
    for h in range(len(nome_datasets)):
    
        # 2. Lendo os datasets  ############################################################################################
        # lendo o dataset
        data = pd.read_csv('Lista03/dataset/'+nome_datasets[h]+'.csv')
        
        # obtendo os padroes e seus respectivos rotulos
        df_x = np.asarray(data.iloc[:,0:-1])
        df_y = np.asarray(data.iloc[:,-1])
        
        
        # 2.1. Criando a tabela para salvar os dados  #################################################
        # criando a tabela que vai acomodar o modelo
        tabela = Tabela_excel()
        tabela.Criar_tabela(nome_tabela='Lista03/arquivos_lista03/'+nome_datasets[h], 
                            folhas=['OLA', 'LCA', 'KNORA-E', 'KNORA-U', 'Arquitetura'], 
                            cabecalho=['acuracy', 'auc', 'fmeasure', 'gmean'], 
                            largura_col=5000)
        # 2.1. End #####################################################################################
        # 2. End ############################################################################################################
        
        # executando os algoritmos x vezes
        for j in range(qtd_execucoes):
            
            # 3. Dividindo os dados para treinamento e teste ################################################################
            # quebrando o dataset sem sobreposicao em 90% para treinamento e 10% para teste  
            skf = StratifiedKFold(df_y, n_folds=qtd_folds)
                
            # tomando os indices para treinamento e teste
            train_index, test_index = next(iter(skf))
                        
            # obtendo os conjuntos de dados para treinamento e teste
            x_train = df_x[train_index]
            y_train = df_y[train_index]
            x_test = df_x[test_index]
            y_test = df_y[test_index]
            # 3. End #########################################################################################################
            
            
            # 4. Gerando o pool de classificadores  ##########################################################################
            # intanciando o classificador
            ensemble = BaggingClassifier(base_estimator=Perceptron(), 
                                            max_samples=qtd_amostras, 
                                            max_features=1.0, 
                                            n_estimators = qtd_modelos)
                    
            # treinando o modelo
            ensemble.fit(x_train, y_train)
            # 4. End  ########################################################################################################
            
            # 5. Instanciando os classificadores ##########################################################
            
            ################################### OLA ########################################################
            executar_modelo('OLA', x_train, y_train, x_test, y_test, ensemble.estimators_, n_vizinhos, nome_datasets, h, j, tabela)
            ################################################################################################
            
            ################################### LCA ########################################################
            executar_modelo('LCA', x_train, y_train, x_test, y_test, ensemble.estimators_, n_vizinhos, nome_datasets, h, j, tabela)
            ################################################################################################
            
            ################################### KNORAE #####################################################
            executar_modelo('KNORAE', x_train, y_train, x_test, y_test, ensemble.estimators_, n_vizinhos, nome_datasets, h, j, tabela)
            ################################################################################################
            
            ################################### KNORAU #####################################################
            executar_modelo('KNORAU', x_train, y_train, x_test, y_test, ensemble.estimators_, n_vizinhos, nome_datasets, h, j, tabela)
            ################################################################################################
            
            ################################### Arquitetura ################################################
            # importando o metodo
            arq = Arquitetura(n_vizinhos)
            # treinando o metodo
            arq.fit(x_train, y_train, x_train, y_train)
            # realizando a previsao
            pred = arq.predict(x_test)
            # printando os resultados
            nome = 'Arquitetura'
            acuracia, auc, f1measure, gmean = printar_resultados(y_test, pred, nome_datasets[h]+'-'+nome+'-['+str(j)+']')
            # escrevendo os resultados obtidos
            tabela.Adicionar_Sheet_Linha(4, j, [acuracia, auc, f1measure, gmean])
            ################################################################################################
            
            # 5. End #########################################################################################
            
if __name__ == "__main__":
    main() 