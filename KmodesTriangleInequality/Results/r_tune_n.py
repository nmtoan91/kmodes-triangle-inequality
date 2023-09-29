from toansttlib import *
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

df = 

datasetIds = ['Dialog','Emotions_MultiTurnDiag','Question_Answer_1','Question_Answer_2','Question_Answer_3','WikiQA']
params = { "text.usetex" : False,"font.family" : "serif"}
plt.rcParams.update(params)
fig, axs = plt.subplots(1,2, figsize=(8, 4))


fig.tight_layout(pad=2.0)
fig.subplots_adjust(bottom=0.3)
ys = []
ys_time=[]
for i in range(6):
    data = pd.read_csv('./FigureDraw/TESTING_epoch/training__home_nmtoan_DATA_CHATBOT_'+datasetIds[i]+'_glue_.csv')
    x = data.epoch.to_numpy()
    y = data.train_loss.to_numpy()
    ys.append(y)
    ys_time.append(data.train_time.to_numpy())
fmta(-1)
DrawSubFigure_SimpleLine(axs[0],x,ys,x_title='epochs',y_title='Loss')
fmta(-1)
DrawSubFigure_SimpleLine(axs[1],x,ys_time,x_title='epochs',y_title='Training time (seconds)')

axs[0].legend(datasetIds,ncol=3,bbox_to_anchor=(1.6, -0.65, 0.5, 0.5))


fileName = '/KmodesTriangleInequality/Results/'+os.path.basename(__file__).replace(".py",".pdf")
#fileName2 = "C:/Users/nmtoa/Dropbox/Apps/Overleaf/Toan_ContextBasedFineTuneGPT2/Figures/" + os.path.basename(__file__).replace(".py",".pdf")
plt.savefig(fileName)
#shutil.copyfile(fileName,fileName2 )

plt.show()

