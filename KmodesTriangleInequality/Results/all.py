from toansttlib import *
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from os import listdir
from os.path import isfile, join

dirname = os.path.dirname(__file__)
basename =os.path.basename(__file__)

mypath = 'RESULT/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)

for filename in onlyfiles:
    if filename.find('_tune_d') >= 0 :
        print(filename)
        parameters = filename.replace('.csv','') .split('_')
        parameter_n = int(parameters[5])
        parameter_d = int(parameters[6])
        parameter_k = int(parameters[7])
        parameter_range = int(parameters[8])
        parameter_sigma = int(parameters[9])/100
        df = pd.read_csv(mypath + filename)
        fig, axs = plt.subplots(1,1)#, figsize=(8, 4))
        x = df['d'].to_numpy()
        ys = np.array([df['Time1'].to_numpy(),df['Time2'].to_numpy()])
        fmta(-1)
        DrawSubFigure_SimpleLine(axs,x,ys,x_title="$d$",y_title='Clustering time (second)' )
        axs.legend(['Baseline','Proposed method'],ncol=1)
        fileName = dirname + '/Charts/' + filename.replace('.csv.csv','.pdf')
        plt.title(f"$n$={parameter_n}; $k$={parameter_k}; range={parameter_range}; $\sigma$={parameter_sigma}")
        plt.tight_layout()
        plt.savefig(fileName)

    if filename.find('_tune_n') >= 0 :
        print(filename)
        parameters = filename.replace('.csv','') .split('_')
        parameter_n = int(parameters[5])
        parameter_d = int(parameters[6])
        parameter_k = int(parameters[7])
        parameter_range = int(parameters[8])
        parameter_sigma = int(parameters[9])/100
        df = pd.read_csv(mypath + filename)
        fig, axs = plt.subplots(1,1)#, figsize=(8, 4))
        x = df['n'].to_numpy()
        ys = np.array([df['Time1'].to_numpy(),df['Time2'].to_numpy()])
        fmta(-1)
        DrawSubFigure_SimpleLine(axs,x,ys,x_title="$n$",y_title='Clustering time (second)' )
        axs.legend(['Baseline','Proposed method'],ncol=1)
        fileName = dirname + '/Charts/' + filename.replace('.csv.csv','.pdf')
        plt.title(f"$d$={parameter_d}; $k$={parameter_k}; range={parameter_range}; $\sigma$={parameter_sigma}")
        plt.tight_layout()
        plt.savefig(fileName)

    if filename.find('_tune_k') >= 0 :
        print(filename)
        parameters = filename.replace('.csv','') .split('_')
        parameter_n = int(parameters[5])
        parameter_d = int(parameters[6])
        parameter_k = int(parameters[7])
        parameter_range = int(parameters[8])
        parameter_sigma = int(parameters[9])/100
        df = pd.read_csv(mypath + filename)
        fig, axs = plt.subplots(1,1)#, figsize=(8, 4))
        x = df['k'].to_numpy()
        ys = np.array([df['Time1'].to_numpy(),df['Time2'].to_numpy()])
        fmta(-1)
        DrawSubFigure_SimpleLine(axs,x,ys,x_title="$k$",y_title='Clustering time (second)' )
        axs.legend(['Baseline','Proposed method'],ncol=1)
        fileName = dirname + '/Charts/' + filename.replace('.csv.csv','.pdf')
        plt.title(f"$n$={parameter_n}; $d$={parameter_d}; range={parameter_range}; $\sigma$={parameter_sigma}")
        plt.tight_layout()
        plt.savefig(fileName)

    if filename.find('_tune_range') >= 0 :
        print(filename)
        parameters = filename.replace('.csv','') .split('_')
        parameter_n = int(parameters[5])
        parameter_d = int(parameters[6])
        parameter_k = int(parameters[7])
        parameter_range = int(parameters[8])
        parameter_sigma = int(parameters[9])/100
        df = pd.read_csv(mypath + filename)
        fig, axs = plt.subplots(1,1)#, figsize=(8, 4))
        x = df['range'].to_numpy()
        ys = np.array([df['Time1'].to_numpy(),df['Time2'].to_numpy()])
        fmta(-1)
        DrawSubFigure_SimpleLine(axs,x,ys,x_title="range",y_title='Clustering time (second)' )
        axs.legend(['Baseline','Proposed method'],ncol=1)
        fileName = dirname + '/Charts/' + filename.replace('.csv.csv','.pdf')
        plt.title(f"$n$={parameter_n}; $d$={parameter_d}; $k$={parameter_k}; $\sigma$={parameter_sigma}")
        plt.tight_layout()
        plt.savefig(fileName)

    if filename.find('_tune_sigma') >= 0 :
        print(filename)
        parameters = filename.replace('.csv','') .split('_')
        parameter_n = int(parameters[5])
        parameter_d = int(parameters[6])
        parameter_k = int(parameters[7])
        parameter_range = int(parameters[8])
        parameter_sigma = int(parameters[9])/100
        df = pd.read_csv(mypath + filename)
        fig, axs = plt.subplots(1,1)#, figsize=(8, 4))
        x = df['sigma'].to_numpy()
        ys = np.array([df['Time1'].to_numpy(),df['Time2'].to_numpy()])
        fmta(-1)
        DrawSubFigure_SimpleLine(axs,x,ys,x_title="$\sigma$",y_title='Clustering time (second)' )
        axs.legend(['Baseline','Proposed method'],ncol=1)
        fileName = dirname + '/Charts/' + filename.replace('.csv.csv','.pdf')
        plt.title(f"$n$={parameter_n}; $d$={parameter_d}; $k$={parameter_k}; range={parameter_range}")
        plt.tight_layout()
        plt.savefig(fileName)
