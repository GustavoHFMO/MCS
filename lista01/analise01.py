'''
Created on 21 de ago de 2018
@author: gusta
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def gerar_boxplot_unico(dataset, nome_algoritmo, metrica):
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


# informacoes a serem plotadas

def gerar_boxplot_metrica_geral(dataset, metrica):
    '''
    metodo para gerar um boxplot do desempenho de todos os classificadores ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    data = []
    
    porcentagens = ['0.5', '0.6', '0.7', '0.8', '0.9', '1']
    
    for i in porcentagens:
        quadro = (i)
        alg1 = pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='bag_dt')[metrica]
        alg2 = pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='rs_dt')[metrica]
        alg3 = pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='bag_p')[metrica]
        alg4 = pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='rs_p')[metrica]
        data.append([quadro+"%", [alg1, alg2, alg3, alg4]])
    
    # list with labels of each square of boxplot
    attemptlist = ['bag_dt','rs_dt','bag_p','rs_p'] 
    
    # colors for the box
    colors = ['pink', 'lightblue', 'lightgreen']
    
    # instantiating each square of boxplot  
    fig, axes = plt.subplots(ncols=len(data), sharey=True)
    fig.subplots_adjust(wspace=0)
    
    # setting the ylabel
    if(metrica=='acuracy'):
        axes[0].set_ylabel('accuracy')
    else:
        axes[0].set_ylabel(metrica)
    
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
        if(i==5):
            ax.legend(box["boxes"], ['BAG+DT', 'RS+DT', 'BAG+P', 'RS+P'], loc='upper right')
        
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
        # auxiliary
        i+=1
    
    plt.suptitle("Dataset: "+dataset)
    plt.show()

def gerar_boxplot_metrica_bgrs(dataset, metrica):
    '''
    metodo para gerar um boxplot do desempenho do bagging e random subspace ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    
    data = []
    
    porcentagens = ['0.5', '0.6', '0.7', '0.8', '0.9', '1']
    
    for i in porcentagens:
        quadro = (i)
        alg1 = np.asarray(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='bag_dt')[metrica])
        alg2 = np.asarray(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='bag_p')[metrica])
        alg3 = np.asarray(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='rs_dt')[metrica])
        alg4 = np.asarray(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='rs_p')[metrica])
        data.append([quadro+"%", [np.concatenate((alg1,alg2)), np.concatenate((alg3,alg4))]])
    
    # list with labels of each square of boxplot
    attemptlist = ['bagging','random subspace'] 
    
    # colors for the box
    colors = ['pink', 'lightblue']
    
    # instantiating each square of boxplot  
    fig, axes = plt.subplots(ncols=len(data), sharey=True)
    fig.subplots_adjust(wspace=0)
    
    # setting the ylabel
    if(metrica=='acuracy'):
        axes[0].set_ylabel('accuracy')
    else:
        axes[0].set_ylabel(metrica)
    
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
        if(i==5):
            ax.legend(box["boxes"], ['Bagging', 'Random Subspace'], loc='upper right')
        
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
        # auxiliary
        i+=1
    
    plt.suptitle("Dataset: "+dataset)
    plt.show()

def gerar_boxplot_metrica_dtp(dataset, metrica):
    '''
    metodo para gerar um boxplot do desempenho do perceptron e decision tree ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    data = []
    
    porcentagens = ['0.5', '0.6', '0.7', '0.8', '0.9', '1']
    
    for i in porcentagens:
        quadro = (i)
        alg1 = np.asarray(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='bag_dt')[metrica])
        alg2 = np.asarray(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='bag_p')[metrica])
        alg3 = np.asarray(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='rs_dt')[metrica])
        alg4 = np.asarray(pd.read_excel(io='arquivos_lista01/'+dataset+'-pct-'+i+'.xls', sheetname='rs_p')[metrica])
        data.append([quadro+"%", [np.concatenate((alg1,alg3)), np.concatenate((alg2,alg4))]])
    
    # list with labels of each square of boxplot
    attemptlist = ['decision tree','perceptron'] 
    
    # colors for the box
    colors = ['pink', 'lightblue']
    
    # instantiating each square of boxplot  
    fig, axes = plt.subplots(ncols=len(data), sharey=True)
    fig.subplots_adjust(wspace=0)
    
    # setting the ylabel
    if(metrica=='acuracy'):
        axes[0].set_ylabel('accuracy')
    else:
        axes[0].set_ylabel(metrica)
    
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
        if(i==5):
            ax.legend(box["boxes"], ['Decision Tree', 'Main'], loc='upper right')
        
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
        # auxiliary
        i+=1
    
    plt.suptitle("Dataset: "+dataset)
    plt.show()

def gerar_boxplot_metrica_bgrs_twodata(metrica):
    '''
    metodo para gerar um boxplot para as duas series contendo o desempenho do bagging e random subspace ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    
    data = []
    
    porcentagens = ['0.5', '0.6', '0.7', '0.8', '0.9', '1']
    
    for i in porcentagens:
        
        if(i=='0.5'):
            quadro = '50'
        elif(i=='0.6'):
            quadro = '60'
        elif(i=='0.7'):
            quadro = '70'
        elif(i=='0.8'):
            quadro = '80'
        elif(i=='0.9'):
            quadro = '90'
        elif(i=='1'):
            quadro = '100'
            
        alg1 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='bag_dt')[metrica])
        alg2 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='bag_p')[metrica])
        alg3 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='rs_dt')[metrica])
        alg4 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='rs_p')[metrica])
        
        alg5 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='bag_dt')[metrica])
        alg6 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='bag_p')[metrica])
        alg7 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='rs_dt')[metrica])
        alg8 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='rs_p')[metrica])
        
        data.append([quadro+"%", [np.concatenate((alg1,alg2,alg5,alg6)), np.concatenate((alg3,alg4,alg7,alg8))]])
        
    # list with labels of each square of boxplot
    attemptlist = ['bagging','random subspace'] 
    
    # colors for the box
    colors = ['pink', 'lightblue']
    
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
    elif(metrica=='auc'):
        axes[0].set_ylabel('AUC')
    
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
        if(i==5):
            ax.legend(box["boxes"], ['Bagging', 'Random Subspace'], loc='upper right')
        
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
        # auxiliary
        i+=1
    
    plt.suptitle("Datasets: kc1 and kc2")
    plt.show()
    
def gerar_boxplot_metrica_dtp_twodata(metrica):
    '''
    metodo para gerar um boxplot para as duas series contendo o desempenho do perceptron e decision tree ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    
    data = []
    
    porcentagens = ['0.5', '0.6', '0.7', '0.8', '0.9', '1']
    
    for i in porcentagens:
        if(i=='0.5'):
            quadro = '50'
        elif(i=='0.6'):
            quadro = '60'
        elif(i=='0.7'):
            quadro = '70'
        elif(i=='0.8'):
            quadro = '80'
        elif(i=='0.9'):
            quadro = '90'
        elif(i=='1'):
            quadro = '100'
        alg1 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='bag_dt')[metrica])
        alg2 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='bag_p')[metrica])
        alg3 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='rs_dt')[metrica])
        alg4 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='rs_p')[metrica])
        
        alg5 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='bag_dt')[metrica])
        alg6 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='bag_p')[metrica])
        alg7 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='rs_dt')[metrica])
        alg8 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='rs_p')[metrica])
        
        data.append([quadro+"%", [np.concatenate((alg1,alg3,alg5,alg7)), np.concatenate((alg2,alg4,alg6,alg8))]])
    
    # list with labels of each square of boxplot
    attemptlist = ['decision tree','perceptron'] 
    
    # colors for the box
    colors = ['pink', 'lightblue']
    
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
    elif(metrica=='auc'):
        axes[0].set_ylabel('AUC')
    
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
        if(i==5):
            ax.legend(box["boxes"], ['Decision Tree', 'Main'], loc='upper right')
        
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
        # auxiliary
        i+=1
    
    plt.suptitle("Datasets: kc1 and kc2")
    plt.show()

def gerar_boxplot_metrica_geral_twodata(metrica):
    '''
    metodo para gerar um boxplot para as duas series contendo o desempenho de todos os classificadores ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    
    data = []
    
    porcentagens = ['0.5', '0.6', '0.7', '0.8', '0.9', '1']
    
    for i in porcentagens:
        if(i=='0.5'):
            quadro = '50'
        elif(i=='0.6'):
            quadro = '60'
        elif(i=='0.7'):
            quadro = '70'
        elif(i=='0.8'):
            quadro = '80'
        elif(i=='0.9'):
            quadro = '90'
        elif(i=='1'):
            quadro = '100'
        alg1 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='bag_dt')[metrica])
        alg2 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='bag_p')[metrica])
        alg3 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='rs_dt')[metrica])
        alg4 = np.asarray(pd.read_excel(io='arquivos_lista01/kc1-pct-'+i+'.xls', sheetname='rs_p')[metrica])
        
        alg5 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='bag_dt')[metrica])
        alg6 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='bag_p')[metrica])
        alg7 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='rs_dt')[metrica])
        alg8 = np.asarray(pd.read_excel(io='arquivos_lista01/kc2-pct-'+i+'.xls', sheetname='rs_p')[metrica])
        
        data.append([quadro+"%", [np.concatenate((alg1,alg5)), np.concatenate((alg3,alg7)), np.concatenate((alg2,alg6)), np.concatenate((alg4,alg8))]])
    
    # list with labels of each square of boxplot
    attemptlist = ['bag_dt','rs_dt','bag_p','rs_p'] 
    
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
    elif(metrica=='auc'):
        axes[0].set_ylabel('AUC')
    
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
        if(i==5):
            ax.legend(box["boxes"], ['BAG+DT', 'RS+DT', 'BAG+P', 'RS+P'], loc='upper right')
        
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
        # auxiliary
        i+=1
    
    plt.suptitle("Datasets: kc1 and kc2")
    plt.show()

metricas = ['acuracy', 'auc', 'fmeasure', 'gmean']

# gerando os boxplots para o relatorio

# boxplots para a primeira analise
gerar_boxplot_metrica_bgrs_twodata(metricas[0])
gerar_boxplot_metrica_bgrs_twodata(metricas[1])
gerar_boxplot_metrica_bgrs_twodata(metricas[2])
gerar_boxplot_metrica_bgrs_twodata(metricas[3])

# boxplots para a segunda analise
gerar_boxplot_metrica_dtp_twodata(metricas[0])
gerar_boxplot_metrica_dtp_twodata(metricas[1])
gerar_boxplot_metrica_dtp_twodata(metricas[2])
gerar_boxplot_metrica_dtp_twodata(metricas[3])

# boxplots para a terceira analise
gerar_boxplot_metrica_geral_twodata(metricas[0])
gerar_boxplot_metrica_geral_twodata(metricas[1])
gerar_boxplot_metrica_geral_twodata(metricas[2])
gerar_boxplot_metrica_geral_twodata(metricas[3])

