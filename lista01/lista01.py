'''
Created on 21 de ago de 2018
@author: gusta
'''

# instanciando todas as libs necessarias
from geradores_tabela.Tabela_excel import Tabela_excel
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.cross_validation import StratifiedKFold
from sklearn.tree.tree import DecisionTreeClassifier
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import pandas as pd
import numpy as np

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
    
    # retornando os resultados
    return acuracia, auc, f1measure, gmean



# 1. Definindo as configuracoes experimentais #######################################################################

pct_trainamento = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
qtd_modelos = 100
qtd_execucoes = 10
qtd_folds = 10
nome_datasets = ['kc1', 'kc2']

# 1. End #############################################################################################################


# for para variar entre os datasets
for h in range(len(nome_datasets)):

    # 2. Leitura arquivos  ###########################################################################################
    
    # lendo o dataset
    data = pd.read_csv('dataset/'+nome_datasets[h]+'.csv')
    
    # obtendo os padroes e seus respectivos rotulos
    df_x = np.asarray(data.iloc[:,0:-1])
    df_y = np.asarray(data.iloc[:,-1])
    
    # 2. End ##############################################################################################################
    
    
    # 3. Etapa de execucao dos classificadores ############################################################################
    
    # for para executar para diferentes tipos de tamanhos de dataset
    for i in range(len(pct_trainamento)):
        
        
        # criando a tabela que vai acomodar o modelo
        tabela = Tabela_excel()
        tabela.Criar_tabela(nome_tabela=nome_datasets[h]+'-pct-'+str(pct_trainamento[i]), 
                        folhas=['bag_dt', 'bag_p', 'rs_dt', 'rs_p'], 
                        cabecalho=['acuracy', 'auc', 'fmeasure', 'gmean'], 
                        largura_col=5000)
        
        # for para executar pela quantidade de execucoes
        for j in range(qtd_execucoes):
        
            # 3.1. Tratando os dados para randomizar as amostras com o cross validation ################################
            
            # quebrando o dataset sem sobreposicao em 90% para treinamento e 10% para teste  
            skf = StratifiedKFold(df_y, n_folds=qtd_folds)
            # tomando o atual index para treinamento e teste
            train_index, test_index = next(iter(skf))
            
            # obtendo os conjuntos de dados para treinamento e teste
            x_train = df_x[train_index]
            y_train = df_y[train_index]
            x_test = df_x[test_index]
            y_test = df_y[test_index]
            
            # 3.1. End ###################################################################################################
            
            
            # 3.2 Instanciando os classificadores  #########################################################################
            
            
            # 3.2.1. Bagging com DecisionTree ############################################################
            
            # numero do modelo na tabela 
            num_model = 0
            
            # modelo
            bg = BaggingClassifier(base_estimator=DecisionTreeClassifier(), 
                                   max_samples = pct_trainamento[i], 
                                   max_features=1.0, 
                                   n_estimators = qtd_modelos)
            # treinando o modelo
            bg.fit(x_train, y_train)
            
            # computando a previsao
            pred = bg.predict(x_test)
            
            # printando os resultados
            acuracia, auc, f1measure, gmean = printar_resultados(y_test, pred, nome_datasets[h]+'-pct-'+str(pct_trainamento[i])+'- Bagging com DecisionTree ['+str(j)+']')
            
            # escrevendo os resultados obtidos
            tabela.Adicionar_Sheet_Linha(num_model, j, [acuracia, auc, f1measure, gmean])
            
            # 3.2.1. End ###################################################################################
            
            
            # 3.2.2. Bagging com Main ################################################################
            # numero do modelo na tabela 
            num_model = 1
            
            # modelo
            bg = BaggingClassifier(base_estimator=Perceptron(), 
                                   max_samples = pct_trainamento[i], 
                                   max_features=1.0, 
                                   n_estimators = qtd_modelos)
            # treinando o modelo
            bg.fit(x_train, y_train)
            
            # computando a previsao
            pred = bg.predict(x_test)
            
            # printando os resultados
            acuracia, auc, f1measure, gmean = printar_resultados(y_test, pred, nome_datasets[h]+'-pct-'+str(pct_trainamento[i])+'- Bagging com Main ['+str(j)+']')
            
            # escrevendo os resultados obtidos
            tabela.Adicionar_Sheet_Linha(num_model, j, [acuracia, auc, f1measure, gmean])
            
            # 3.2.2. End ###################################################################################
            
            
            
            # 3.2.3. Random Subspace com DecisionTree ############################################################
            # numero do modelo na tabela 
            num_model = 2
            
            # modelo
            bg = BaggingClassifier(base_estimator=DecisionTreeClassifier(), 
                                   max_samples = pct_trainamento[i], 
                                   max_features=0.5, 
                                   n_estimators = qtd_modelos)
            # treinando o modelo
            bg.fit(x_train, y_train)
            
            # computando a previsao
            pred = bg.predict(x_test)
            
            # printando os resultados
            acuracia, auc, f1measure, gmean = printar_resultados(y_test, pred, nome_datasets[h]+'-pct-'+str(pct_trainamento[i])+'- Random Subspace com DecisionTree ['+str(j)+']')
            
            # escrevendo os resultados obtidos
            tabela.Adicionar_Sheet_Linha(num_model, j, [acuracia, auc, f1measure, gmean])
            
            # 3.2.3. End ###################################################################################
            
            
            # 3.2.4. Bagging com Main ################################################################
            # numero do modelo na tabela 
            num_model = 3
            
            # modelo
            bg = BaggingClassifier(base_estimator=Perceptron(), 
                                   max_samples = pct_trainamento[i], 
                                   max_features=0.5, 
                                   n_estimators = qtd_modelos)
            # treinando o modelo
            bg.fit(x_train, y_train)
            
            # computando a previsao
            pred = bg.predict(x_test)
            
            # printando os resultados
            acuracia, auc, f1measure, gmean = printar_resultados(y_test, pred, nome_datasets[h]+'-pct-'+str(pct_trainamento[i])+'- Random Subspace com Main ['+str(j)+']')
            
            # escrevendo os resultados obtidos
            tabela.Adicionar_Sheet_Linha(num_model, j, [acuracia, auc, f1measure, gmean])
            
            # 3.2.4. End ###################################################################################
            
            # 3.2 End ######################################################################################################
    
    
    # 3. End ##################################################################################################################





