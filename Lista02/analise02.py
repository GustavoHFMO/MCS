#-*- coding: utf-8 -*-
'''
Created on 21 de ago de 2018
@author: gusta
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def gerar_boxplot_validacao(data_name, metrica):
    '''
    metodo para gerar um boxplot para as duas series contendo o desempenho de todos os classificadores ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    data = []
    labels = []
    validacao = ['completa', 'faceis', 'dificeis']
    
    for i in validacao:
        if(i==validacao[0]):
            quadro = 'Validação completa'
            labels.append(quadro)
        elif(i==validacao[1]):
            quadro = 'Observações fáceis'
            labels.append(quadro)
        elif(i==validacao[2]):
            quadro = 'Observações difíceis'
            labels.append(quadro)
            
        alg1 = np.asarray(pd.read_excel(io='arquivos_lista02/'+data_name+'-val-'+i+'.xls', sheet_name='Bagging+REP')[metrica])
        alg2 = np.asarray(pd.read_excel(io='arquivos_lista02/'+data_name+'-val-'+i+'.xls', sheet_name='Bagging+OGOB')[metrica])
        
        data.append([np.concatenate((alg1,alg2))])
    
    # colors for the box
    colors = ['pink', 'lightblue', 'lightgreen']
    
    # instantiating each square of boxplot  
    fig, axes = plt.subplots(1, sharey=True)
    fig.subplots_adjust(wspace=0)
    
    # creating the boxplot
    box = axes.boxplot(data, labels=labels, patch_artist=True)
    axes.yaxis.grid(True, alpha=0.1)
    
    # defining the collor for each boxplot
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
            
    # setting the ylabel
    if(metrica=='acuracy'):
        axes.set_ylabel('Accuracy')
    elif(metrica=='fmeasure'):
        axes.set_ylabel('F-measure')
    elif(metrica=='gmean'):
        axes.set_ylabel('G-mean')
    elif(metrica=='disagreement'):
        axes.set_ylabel('Disagreement Measure')
    elif(metrica=='q statistic'):
        axes.set_ylabel('Q statistic')
    elif(metrica=='qtd modelos'):
        axes.set_ylabel('Quantidade de modelos')
    
    
    plt.suptitle("Dataset: " +data_name)
    plt.show()

def gerar_boxplot_metrica_geral_onedata(data_name, metrica):
    '''
    metodo para gerar um boxplot para as duas series contendo o desempenho de todos os classificadores ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    data = []
    validacao = ['completa', 'faceis', 'dificeis']
    
    for i in validacao:
        if(i==validacao[0]):
            quadro = 'Validação completa'
        elif(i==validacao[1]):
            quadro = 'Observações fáceis'
        elif(i==validacao[2]):
            quadro = 'Observações difíceis'
            
        alg1 = np.asarray(pd.read_excel(io='arquivos_lista02/'+data_name+'-val-'+i+'.xls', sheet_name='Bagging+REP')[metrica])
        alg2 = np.asarray(pd.read_excel(io='arquivos_lista02/'+data_name+'-val-'+i+'.xls', sheet_name='Bagging+OGOB')[metrica])
        
        data.append([quadro, [alg1, alg2]])
    
    # list with labels of each square of boxplot
    attemptlist = ['Bagging+REP','Bagging+OGOB'] 
    
    # colors for the box
    colors = ['pink', 'lightblue', 'lightgreen']
    
    # instantiating each square of boxplot  
    fig, axes = plt.subplots(ncols=len(data), sharey=True)
    fig.subplots_adjust(wspace=0)
    
    # setting the ylabel
    if(metrica=='acuracy'):
        axes[0].set_ylabel('Accuracy')
    elif(metrica=='fmeasure'):
        axes[0].set_ylabel('F-measure')
    elif(metrica=='gmean'):
        axes[0].set_ylabel('G-mean')
    elif(metrica=='disagreement'):
        axes[0].set_ylabel('Disagreement Measure')
    elif(metrica=='q statistic'):
        axes[0].set_ylabel('Q statistic')
    elif(metrica=='qtd modelos'):
        axes[0].set_ylabel('Quantidade de modelos')
    
    i=0
    # creating a figure with each boxplot instanted in the previous line
    for ax, d in zip(axes,data):  
        
        # creating the current boxplot with its respective label   
        box = ax.boxplot([d[1][attemptlist.index(attempt)] for attempt in attemptlist], patch_artist=True)
        
        # for each square and boxplot define the current label
        #ax.set_xticklabels(attemptlist, rotation=90, ha='right')
        ax.set(xlabel=d[0])
        
        # defining the collor for each boxplot
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # plotting the legend
        if(i==2):
            ax.legend(box["boxes"], ['Bagging+REP','Bagging+OGOB'], loc='upper right')
        
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
        # auxiliary
        i+=1
    
    plt.suptitle("Dataset: " +data_name)
    plt.show()

def gerar_boxplot_metrica_geral_twodata(metrica):
    '''
    metodo para gerar um boxplot para as duas series contendo o desempenho de todos os classificadores ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    
    data = []
    
    validacao = ['completa', 'faceis', 'dificeis']
    
    for i in validacao:
        if(i==validacao[0]):
            quadro = 'Validação completa'
        elif(i==validacao[1]):
            quadro = 'Observações fáceis'
        elif(i==validacao[2]):
            quadro = 'Observações difíceis'
            
        alg1 = np.asarray(pd.read_excel(io='arquivos_lista02/kc1-val-'+i+'.xls', sheet_name='Bagging+REP')[metrica])
        alg2 = np.asarray(pd.read_excel(io='arquivos_lista02/kc1-val-'+i+'.xls', sheet_name='Bagging+OGOB')[metrica])
        
        alg5 = np.asarray(pd.read_excel(io='arquivos_lista02/kc2-val-'+i+'.xls', sheet_name='Bagging+REP')[metrica])
        alg6 = np.asarray(pd.read_excel(io='arquivos_lista02/kc2-val-'+i+'.xls', sheet_name='Bagging+OGOB')[metrica])
        
        data.append([quadro, [np.concatenate((alg1,alg5)), np.concatenate((alg2,alg6))]])
    
    # list with labels of each square of boxplot    
    attemptlist = ['Bagging+REP','Bagging+OGOB'] 
    
    # colors for the box
    colors = ['pink', 'lightblue', 'lightgreen']
    
    # instantiating each square of boxplot  
    fig, axes = plt.subplots(ncols=len(data), sharey=True)
    fig.subplots_adjust(wspace=0)
    
    # setting the ylabel
    if(metrica=='acuracy'):
        axes[0].set_ylabel('Accuracy')
    elif(metrica=='fmeasure'):
        axes[0].set_ylabel('F-measure')
    elif(metrica=='gmean'):
        axes[0].set_ylabel('G-mean')
    elif(metrica=='disagreement'):
        axes[0].set_ylabel('Disagreement Measure')
    elif(metrica=='q statistic'):
        axes[0].set_ylabel('Q statistic')
    elif(metrica=='qtd modelos'):
        axes[0].set_ylabel('Quantidade de modelos')
    
    i=0
    # creating a figure with each boxplot instanted in the previous line
    for ax, d in zip(axes,data):  
        
        # creating the current boxplot with its respective label   
        box = ax.boxplot([d[1][attemptlist.index(attempt)] for attempt in attemptlist], patch_artist=True)
        
        # for each square and boxplot define the current label
        #ax.set_xticklabels(attemptlist, rotation=90, ha='right')
        ax.set(xlabel=d[0])
        
        # defining the collor for each boxplot
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # plotting the legend
        if(i==2):
            ax.legend(box["boxes"], ['REP','OGOB'], loc='upper right')
        
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
        # auxiliary
        i+=1
    
    plt.suptitle("Datasets: kc1 and kc2")
    plt.show()

metricas = ['qtd modelos', 'q statistic', 'disagreement', 'acuracy', 'auc', 'fmeasure', 'gmean']
data = ['kc1', 'kc2']


# gerando os boxplots para o relatorio
#gerar_boxplot_metrica_geral_twodata(metricas[1])

#gerar_boxplot_metrica_geral_onedata(data[0], metricas[1])
#gerar_boxplot_metrica_geral_onedata(data[1], metricas[2])
#gerar_boxplot_metrica_geral_onedata(data[1], metricas[3])


print(pd.read_excel("../Lista02/arquivos_lista02/geral.xlsx"))

