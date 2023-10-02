from toansttlib import *
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

df = pd.read_csv('KmodesTriangleInequality/Results/'+os.path.basename(__file__).replace(".py",".csv"))

#datasetIds = ['Dialog','Emotions_MultiTurnDiag','Question_Answer_1','Question_Answer_2','Question_Answer_3','WikiQA']
#params = { "text.usetex" : False,"font.family" : "serif"}
#plt.rcParams.update(params)


fig, axs = plt.subplots(1,2, figsize=(8, 4))
x = df['d'].to_numpy()
ys = np.array([df['Time1'].to_numpy(),df['Time2'].to_numpy()])
fmta(-1)
DrawSubFigure_SimpleLine(axs[0],x,ys,x_title='d',y_title='Clustering time (second)' )
axs[0].legend(['Baseline','Proposed method'],ncol=1)




fileName = 'KmodesTriangleInequality/Results/'+os.path.basename(__file__).replace(".py",".pdf")
#fileName2 = "C:/Users/nmtoa/Dropbox/Apps/Overleaf/Toan_ContextBasedFineTuneGPT2/Figures/" + os.path.basename(__file__).replace(".py",".pdf")
plt.savefig(fileName)
#shutil.copyfile(fileName,fileName2 )

plt.show()

