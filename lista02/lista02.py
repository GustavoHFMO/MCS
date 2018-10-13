'''
Created on 21 de ago de 2018
@author: gusta
'''

# instanciando todas as libs necessarias
from deslib.util.diversity import Q_statistic, disagreement_measure
from lista02.geradores_tabela.Tabela_excel import Tabela_excel
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.linear_model.perceptron import Perceptron
from sklearn.cross_validation import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import pandas as pd
import numpy as np
import copy

def printar_resultados(y_test, pred, ensemble, nome_modelo):
    '''
    metodo para printar os resultados de cada modelo
    :param: y_test: dados correspondentes a saida de teste
    :param: pred: dados correspondentes a previsao do modelo
    :return: retorna as metricas: acuracia, auc, f1score e gmean
    '''
    
    # computando as metricas para os dados recebidos
    qtd_modelos = len(ensemble.estimators_)
    acuracia = metrics.accuracy_score(y_test, pred)
    auc = metrics.roc_auc_score(y_test, pred)
    f1measure = metrics.f1_score(y_test, pred, average='binary')
    gmean = geometric_mean_score(y_test, pred, average='binary')
    
    # calculando o desempenho
    print('\n'+nome_modelo)
    print("qtd modelos:", qtd_modelos)
    print("taxa de acerto:", acuracia)
    print("AUC:", auc)
    print("f-measure:", f1measure)
    print("g-mean:", gmean)
    
    # retornando os resultados
    return qtd_modelos, acuracia, auc, f1measure, gmean



# 1. Definindo as configuracoes experimentais #######################################################################

def validacaoCompleta(x, y):
    '''
    Metodo para retornar um subconjunto de validacao dado um conjunto maior
    :param: x: padroes dos dados
    :param: y: respectivos rotulos
    :return: x_new, y_new: 
    '''
    
    x_new = x
    y_new = y
    
    return x_new, y_new

def validacaoInstanciasFaceis(x, y, n_vizinhos):
    '''
    Metodo para retornar um subconjunto de validacao apenas com as instacias faceis
    :param: x: padroes dos dados
    :param: y: respectivos rotulos
    :return: x_new, y_new: 
    '''
    
    # computando as dificulades para cada instancia
    dificuldades = kDN(x, y, n_vizinhos)
    
    # variaveis para salvar as novas instancias
    x_new = []
    y_new = []
    
    # salvando apenas as instancias faceis
    for i in range(len(dificuldades)):
        if(dificuldades[i] < 0.5):
            x_new.append(x[i])
            y_new.append(y[i])
            
    
    return np.asarray(x_new), np.asarray(y_new)

def validacaoInstanciasDificeis(x, y, n_vizinhos):
    '''
    Metodo para retornar um subconjunto de validacao apenas com as instacias dificeis
    :param: x: padroes dos dados
    :param: y: respectivos rotulos
    :return: x_new, y_new: 
    '''
    
    # computando as dificulades para cada instancia
    dificuldades = kDN(x, y, n_vizinhos)
    
    # variaveis para salvar as novas instancias
    x_new = []
    y_new = []
    
    # salvando apenas as instancias faceis
    for i in range(len(dificuldades)):
        if(dificuldades[i] > 0.5):
            x_new.append(x[i])
            y_new.append(y[i])
            
    
    return np.asarray(x_new), np.asarray(y_new)

def kDN(x, y, n_vizinhos):
    '''
    Metodo para computar o grau de dificuldade de cada observacao em um conjunto de dados
    :param: x: padroes dos dados
    :param: y: respectivos rotulos
    :return: dificuldades: vetor com a probabilidade de cada instancia 
    '''
    
    # instanciando os vizinhos mais proximos
    nbrs = NearestNeighbors(n_neighbors=n_vizinhos+1, algorithm='ball_tree').fit(x)
    
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
        dificuldades.append(cont/(n_vizinhos+1))

    
    return dificuldades

def REP(x_val, y_val, ensemble):
    '''
    Metodo ReduceErrorPrunning para fazer uma poda na quantidade de modelos
    :param: x_val: dados com atributos que serao usados para a poda 
    :param: y_val: dados com rotulos que serao usados para a poda
    :return: ensemble: novo ensemble com os novos estimadores
    '''
    
    # variavel para armazenar o desempenho dos estimadores
    desempenho = [0] * len(ensemble.estimators_)
    
    # obtendo o desempenho dos estimadores para o conjunto de validacao
    for i in range(len(ensemble.estimators_)):
        pred = ensemble.estimators_[i].predict(x_val)
        desempenho[i] = metrics.accuracy_score(y_val, pred)
        
    # ordenando os classificadores com base no seu desempenho do pior para o melhor
    ordem = sorted(range(len(desempenho)), key=lambda k: desempenho[k])
    
    # criando um novo ensemble
    new_ensemble = copy.deepcopy(ensemble)
    new_ensemble.estimators_ = []
    
    # adicionando o melhor classificador ao novo ensemble
    new_ensemble.estimators_.append(ensemble.estimators_[ordem[-1]])
    
    # iniciando o erro do ensemble
    pred = new_ensemble.predict(x_val)
    erro_inicial = metrics.accuracy_score(y_val, pred)
    
    # adicionando iterativamente os classificadores
    for i in range(len(ensemble.estimators_)):
        
        # excecao para nao adicionar o melhor classificador de novo
        if(i != ordem[-1]):
            
            # adicionando um classificador ao novo pool
            new_ensemble.estimators_.append(ensemble.estimators_[ordem[-i]])

            # computando o erro do ensemble
            pred = new_ensemble.predict(x_val)
            erro = metrics.accuracy_score(y_val, pred)
            
            # adicionando se o erro diminui
            if(erro <= erro_inicial):
                erro_inicial = erro
            else:
                del new_ensemble.estimators_[-1]
            
    return new_ensemble

def OGOB(x_val, y_val, ensemble):
    '''
    Metodo ReduceErrorPrunning para fazer uma poda na quantidade de modelos
    :param: x_val: dados com atributos que serao usados para a poda 
    :param: y_val: dados com rotulos que serao usados para a poda
    :return: ensemble: novo ensemble com os novos estimadores
    '''
    
    # variavel para armazenar o desempenho dos estimadores
    desempenho = [0] * len(ensemble.estimators_)
    
    # obtendo o desempenho dos estimadores para o conjunto de validacao
    for i in range(len(ensemble.estimators_)):
        pred = ensemble.estimators_[i].predict(x_val)
        desempenho[i] = metrics.accuracy_score(y_val, pred)
        
    # ordenando os classificadores com base no seu desempenho do pior para o melhor
    ordem = sorted(range(len(desempenho)), key=lambda k: desempenho[k])
    
    # criando um novo ensemble
    new_ensemble = copy.deepcopy(ensemble)
    new_ensemble.estimators_ = []
    
    # adicionando o melhor classificador ao novo ensemble
    new_ensemble.estimators_.append(ensemble.estimators_[ordem[-1]])
    
    # apagando o melhor classificador do pool inicial
    del ensemble.estimators_[ordem[-1]]
    
    # iniciando o erro do ensemble
    pred = new_ensemble.predict(x_val)
    erro_inicial = metrics.accuracy_score(y_val, pred)
    
    # computando o complemento dos estimadores em relacao ao ensemble
    diversidade = []
    for i in range(len(ensemble.estimators_)):
        predc = ensemble.estimators_[i].predict(x_val)
        prede = new_ensemble.predict(x_val)
        div = np.mean(predc != prede) * 100
        diversidade.append(div)

    # ordenando os modelos da maior para a pior diversidade
    ordem = sorted(range(len(diversidade)), key=lambda k: diversidade[k])
    
    # adicionando iterativamente os classificadores
    for i in range(len(diversidade)):
        
        # adicionando um classificador ao novo pool
        new_ensemble.estimators_.append(ensemble.estimators_[ordem[-i]])

        # computando o erro do ensemble
        pred = new_ensemble.predict(x_val)
        erro = metrics.accuracy_score(y_val, pred)
            
        # adicionando se o erro diminui
        if(erro <= erro_inicial):
            erro_inicial = erro
        else:
            del new_ensemble.estimators_[-1]

                
    return new_ensemble

def MedidasDiversidade(medida, x_val, y_val, ensemble):
    '''
    Metodo para calcular a diversidade de um ensemble utilizando a estatistica Q 
    :param: x_val: dados com atributos que serao usados para a poda 
    :param: y_val: dados com rotulos que serao usados para a poda
    :return: Q: diversidade do ensemble
    '''
    
    L = len(ensemble.estimators_)
    Q = 0
    
    for i in range(L-1):
        for k in range(i+1, L):
            y_pred0 = ensemble.estimators_[i].predict(x_val)
            y_pred1 = ensemble.estimators_[k].predict(x_val)
                
            if(medida == 'q'):
                Q += Q_statistic(y_val, y_pred0, y_pred1)
            elif(medida == 'disagreement'):
                Q += disagreement_measure(y_val, y_pred0, y_pred1)
            
    Q = (2/(L*(L-1)))*Q
    
    return Q





# 1. Definindo variaveis para o experimento #########################################################################

qtd_modelos = 100
qtd_execucoes = 30
qtd_amostras = 0.9
qtd_folds = 10
n_vizinhos = 7
validacao = ['completa', 'faceis', 'dificeis']
nome_datasets = ['kc1', 'kc2']

# 1. End ############################################################################################################

# for para variar entre os datasets
for h in range(len(nome_datasets)):
    
    # 2. Lendo os datasets  ############################################################################################
    
    # lendo o dataset
    data = pd.read_csv('../dataset/'+nome_datasets[h]+'.csv')
    
    # obtendo os padroes e seus respectivos rotulos
    df_x = np.asarray(data.iloc[:,0:-1])
    df_y = np.asarray(data.iloc[:,-1])
    
    # 2. End ############################################################################################################
    
    # 3. Obtendo a janela de validacao ######################################################################################
    
    # obtendo a janela de validacao
    for k in range(len(validacao)):
        
        # 3.1. Criando a tabela para salvar os dados  #################################################
            
        # criando a tabela que vai acomodar o modelo
        tabela = Tabela_excel()
        tabela.Criar_tabela(nome_tabela=nome_datasets[h]+'-val-'+validacao[k], 
                            folhas=['Bagging+REP', 'Bagging+OGOB'], 
                            cabecalho=['qtd modelos', 'q statistic', 'disagreement', 'acuracy', 'auc', 'fmeasure', 'gmean'], 
                            largura_col=5000)
        
        # 3.1. End #####################################################################################
            
        # executando os algoritmos x vezes
        for j in range(qtd_execucoes):
            
            # 3.2. Tratando os dados para randomizar as amostras com o cross validation #############################
        
            # quebrando o dataset sem sobreposicao em 90% para treinamento e 10% para teste  
            skf = StratifiedKFold(df_y, n_folds=qtd_folds)
            
            # tomando os indices para treinamento e teste
            train_index, test_index = next(iter(skf))
                    
            # obtendo os conjuntos de dados para treinamento e teste
            x_train = df_x[train_index]
            y_train = df_y[train_index]
            x_test = df_x[test_index]
            y_test = df_y[test_index]
            
            # 3.2. End ###############################################################################################
            
            # 3.3. Obtendo a janela de validacao #####################################################################
        
            if(validacao[k] == validacao[0]):
                x_val, y_val = validacaoCompleta(x_train, y_train)
            elif(validacao[k] == validacao[1]):
                x_val, y_val = validacaoInstanciasFaceis(x_train, y_train, n_vizinhos)
            elif(validacao[k] == validacao[2]):
                x_val, y_val = validacaoInstanciasDificeis(x_train, y_train, n_vizinhos)
        
            # 3.3. End ################################################################################################
        
            # 3.4. Instanciando os classificadores ##########################################################
                
            ########## instanciando o modelo Bagging+REP ###########################################
            # definindo o numero do modelo na tabela
            num_model = 0
            
            # intanciando o classificador
            ensemble = BaggingClassifier(base_estimator=Perceptron(), 
                                        max_samples=qtd_amostras, 
                                        max_features=1.0, 
                                        n_estimators = qtd_modelos)
                
            # treinando o modelo
            ensemble.fit(x_train, y_train)
                
            # realizando a poda 
            ensemble = REP(x_val, y_val, ensemble)
                        
            # computando a previsao
            pred = ensemble.predict(x_test)
                        
            # computando a diversidade do ensemble
            q_statistic = MedidasDiversidade('q', x_val, y_val, ensemble)
            double_fault = MedidasDiversidade('disagreement', x_val, y_val, ensemble)
                
            # printando os resultados
            qtd_modelos, acuracia, auc, f1measure, gmean = printar_resultados(y_test, pred, ensemble, nome_datasets[h]+'-Bagging-REP-'+validacao[k]+'['+str(j)+']')
            
            # escrevendo os resultados obtidos
            tabela.Adicionar_Sheet_Linha(num_model, j, [qtd_modelos, q_statistic, double_fault, acuracia, auc, f1measure, gmean])
            ###########################################################################################
                
                
            ########## instanciando o modelo Bagging+OGOB ###########################################
            # definindo o numero do modelo na tabela
            num_model = 1
            
            # intanciando o classificador
            ensemble = BaggingClassifier(base_estimator=Perceptron(), 
                                        max_samples=qtd_amostras,
                                        max_features=1.0, 
                                        n_estimators = qtd_modelos)
                
            # treinando o modelo
            ensemble.fit(x_train, y_train)
                
            # realizando a poda 
            ensemble = OGOB(x_val, y_val, ensemble)
                        
            # computando a previsao
            pred = ensemble.predict(x_test)
                        
            # computando a diversidade do ensemble
            q_statistic = MedidasDiversidade('q', x_val, y_val, ensemble)
            double_fault = MedidasDiversidade('disagreement', x_val, y_val, ensemble)
                
            # printando os resultados
            qtd_modelos, acuracia, auc, f1measure, gmean = printar_resultados(y_test, pred, ensemble, nome_datasets[h]+'-Bagging-REP-'+validacao[k]+'['+str(j)+']')
            
            # escrevendo os resultados obtidos
            tabela.Adicionar_Sheet_Linha(num_model, j, [qtd_modelos, q_statistic, double_fault, acuracia, auc, f1measure, gmean])
            ###########################################################################################
                
            # 3.4. End #####################################################################################
        
    # 3. End ############################################################################################################


