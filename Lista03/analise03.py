#-*- coding: utf-8 -*-
'''
Created on 21 de ago de 2018
@author: gusta
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def gerar_boxplot_geral(data_name, metrica):
    '''
    metodo para gerar um boxplot para as duas series contendo o desempenho de todos os classificadores ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    data = []
            
    alg1 = np.asarray(pd.read_excel(io='arquivos_lista03/'+data_name+'.xls', sheetname='OLA')[metrica])
    data.append(alg1)
    alg2 = np.asarray(pd.read_excel(io='arquivos_lista03/'+data_name+'.xls', sheetname='LCA')[metrica])
    data.append(alg2)
    alg3 = np.asarray(pd.read_excel(io='arquivos_lista03/'+data_name+'.xls', sheetname='KNORA-E')[metrica])
    data.append(alg3)
    alg4 = np.asarray(pd.read_excel(io='arquivos_lista03/'+data_name+'.xls', sheetname='KNORA-U')[metrica])
    data.append(alg4)
    alg5 = np.asarray(pd.read_excel(io='arquivos_lista03/'+data_name+'.xls', sheetname='Arquitetura')[metrica])
    data.append(alg5)
        
    # colors for the box
    colors = ['pink', 'lightblue', 'lightgreen', 'blue', 'green']
    
    # instantiating each square of boxplot  
    fig, axes = plt.subplots(1, sharey=True)
    fig.subplots_adjust(wspace=0)
    
    # creating the boxplot
    box = axes.boxplot(data, patch_artist=True)
    axes.yaxis.grid(True, alpha=0.1)
    
    # defining the collor for each boxplot
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # addig the legend
    axes.legend(box["boxes"], ['OLA', 'LCA', 'KNORA-E', 'KNORA-U', 'Arquitetura'], loc='lower right')
            
    # setting the ylabel
    if(metrica=='acuracy'):
        axes.set_ylabel('Accuracy')
    elif(metrica=='auc'):
        axes.set_ylabel('AUC')
    elif(metrica=='fmeasure'):
        axes.set_ylabel('F-measure')
    elif(metrica=='gmean'):
        axes.set_ylabel('G-mean')
    
    plt.suptitle("Dataset: " +data_name)
    plt.show()

def main():
    metricas = ['acuracy', 'auc', 'fmeasure', 'gmean']
    data = ['kc1', 'kc2']
    
    m = 3
    # gerando os boxplots para o relatorio
    gerar_boxplot_geral(data[0], metricas[m])
    gerar_boxplot_geral(data[1], metricas[m])

if __name__ == "__main__":
    main()

