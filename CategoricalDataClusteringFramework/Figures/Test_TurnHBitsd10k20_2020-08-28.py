import pandas as pd
import matplotlib.pyplot as plt
markers = ['o','s','*','x','.','d','^','+','_','.']
lines = ['-','--','-.',':']
def fmt(i):
    return markers[i%len(markers)] + lines[i%len(lines)]

from_ =11
plt.figure(1)
df = pd.read_excel (r'Figures/Test_TurnHBitsd10k20_2020-08-28.xlsx', sheet_name='Purity')
x = list(df['Unnamed: 0'])
for i in range(from_,16):
    v = 2**i
    ax = plt.plot(x,df[str(v)], fmt(i),label='n=' + str(v))
plt.xlabel("Number of hash functions ")
plt.ylabel('Purity')
plt.title('[d=10;k=20;range=20;rate=0.1]')
plt.legend(loc='best')
filename = "Test_TurnHBitsd10k20_2020-08-28_purity.pdf"
plt.savefig("Figures/Figures/"+filename)
plt.savefig("D:/Dropbox/PHD/FIGURES/" + filename)

plt.figure(2)
df = pd.read_excel (r'Figures/Test_TurnHBitsd10k20_2020-08-28.xlsx', sheet_name='TIME_CLUS')
x = list(df['Unnamed: 0'])
for i in range(from_,16):
    v = 2**i
    ax = plt.plot(x,df[str(v)], fmt(i),label='n=' + str(v))
plt.xlabel("Number of hash functions ")
plt.ylabel('Clustering time (second)')
plt.title('[d=10;k=20;range=20;rate=0.1]')
plt.legend(loc='best')
filename = "Test_TurnHBitsd10k20_2020.pdf"
plt.savefig("Figures/Figures/"+filename)
plt.savefig("D:/Dropbox/PHD/FIGURES/" + filename)




plt.show()