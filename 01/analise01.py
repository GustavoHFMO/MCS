'''
Created on 21 de ago de 2018
@author: gusta
'''

import matplotlib.pyplot as plt
import pandas as pd


def gerar_plot(dataset, nome_algoritmo, metrica):
    '''
    metodo para gerar um boxplot do desempenho do classificador ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: nome_algoritmo: nome do algoritmo a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''

    # lendo os arquivos 
    data = []
    data.append(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-0.5.xls', sheetname=nome_algoritmo)[metrica])
    data.append(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-0.6.xls', sheetname=nome_algoritmo)[metrica])
    data.append(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-0.7.xls', sheetname=nome_algoritmo)[metrica])
    data.append(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-0.8.xls', sheetname=nome_algoritmo)[metrica])
    data.append(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-0.9.xls', sheetname=nome_algoritmo)[metrica])
    data.append(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-1.xls', sheetname=nome_algoritmo)[metrica])
    
    # plotando o boxplot
    models = ['0.5', '0.6', '0.7', '0.8', '0.9', '1',]
    plt.boxplot(data, labels=models)
    plt.title(nome_algoritmo)
    plt.ylabel(metrica)
    plt.xlabel('% training dataset')
    plt.legend()
    plt.show()


# plotando os resultados do modelo
'''
datasets = [kc1, kc2]
algoritmos = ['bag_dt', 'bag_p', 'rs_dt', 'rs_p']
metricas = ['acuracy', 'auc', 'fmeasure', 'gmean']
'''
gerar_plot(dataset='kc2',
           nome_algoritmo='bag_dt',
           metrica='acuracy')


