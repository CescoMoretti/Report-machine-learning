import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sea
from sklearn.model_selection import train_test_split

#dataset: https://archive.ics.uci.edu/ml/datasets/SkillCraft1+Master+Table+Dataset

df = pd.read_csv('dataset/SkillCraft1_Dataset.csv', na_values=["?"])

print(df.shape)
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())

#controllo se ci sono dei NaN
print(df.isnull().sum())

#grafico distribuzione dei nan per lega
y = df[df['Age'].isnull()]
z = df[df['HoursPerWeek'].isnull()]
k = df[df['TotalHours'].isnull()]
 
bins = np.linspace(0, 9, 15)
plt.hist([y['LeagueIndex'], z['LeagueIndex'], k['LeagueIndex']],
          bins, 
          label=['Age', 'HoursPerWeek', 'TotalHours'])
plt.title("Distribuzione elementi nulli")
plt.xlabel('Indice della lega')
plt.xticks(range(1, 9))
plt.legend(loc='upper left')
plt.show()

#controllo quanti nulli ci sono nella lega 8
x = df.loc[df['LeagueIndex'] == 8]
print(x) #--> queste colonne sono nulle per tutti i valori della classe 8

#guardo la correlazione di queste colonne con LeagueIndex

cor_mat = df.corr()

sea.set(font_scale=0.8)
h = sea.heatmap(cor_mat, annot=True, fmt='.3f')
h.set_title("Matrice di correlazione - tutte le feature", size=12)
plt.setp(h.get_xticklabels(), rotation=35, ha="right",
         rotation_mode="anchor")


plt.show()

#le colonne Age, HoursPerWeek e TotalHours sono nulle in tutte le righe della lega 8 e in più hanno dei livelli di correlazione con
# LeagueIndex bassi, quindi decido di scartare le colonne. Tolgo anche GameId perchè non lo ritengo fuzionale all'algoritmo

df = df.drop(columns=['Age', 'HoursPerWeek', 'TotalHours']) 



print(df.isnull().sum())
#Non ci sono altri NAN

#Guardo che tipo di relazione hanno le features tra di loro

sea.set(font_scale=0.5)
p = sea.pairplot(df[['LeagueIndex','APM','SelectByHotkeys', 'GapBetweenPACs','NumberOfPACs','ActionLatency',
                #'MinimapAttacks','MinimapRightClicks','AssignToHotkeys',
                #'ActionsInPAC','TotalMapExplored','WorkersMade','UniqueUnitsMade', 'UniqueHotkeys',
                #'ComplexUnitsMade','ComplexAbilitiesUsed'
                ]],
                 kind="reg",
                 diag_kind='kde',
                 plot_kws=dict(scatter_kws=dict(s=0.2),
                 line_kws={'color': 'red'}  )) 
p.set(xticklabels=[])
p.set(yticklabels=[])
plt.show() 
#plot della riga prima riga del pairplot
sea.set(font_scale=0.5)
p1 = sea.pairplot(data=df, y_vars=['LeagueIndex'],
                  x_vars=['APM','SelectByHotkeys','AssignToHotkeys','UniqueHotkeys','MinimapAttacks'],
                  hue='LeagueIndex', 
                  palette='crest')
p1.savefig('p1.png', dpi=300)
plt.close(p1.fig)

p2 = sea.pairplot(data=df, y_vars=['LeagueIndex'], x_vars=['MinimapRightClicks','NumberOfPACs','GapBetweenPACs','ActionLatency','ActionsInPAC',], hue='LeagueIndex', palette='crest')
p2._legend.remove()
p2.savefig('p2.png', dpi=300)
plt.close(p2.fig)

p3 = sea.pairplot(data=df, y_vars=['LeagueIndex'], x_vars=['TotalMapExplored','WorkersMade','UniqueUnitsMade','ComplexUnitsMade','ComplexAbilitiesUsed'], hue='LeagueIndex', palette='crest')
p3._legend.remove()
p3.savefig('p3.png', dpi=300)
plt.close(p3.fig)

f, axes = plt.subplots(3, 1)

axes[0].imshow(mpimg.imread('pairplot_parziale/p1.png'))
axes[1].imshow(mpimg.imread('pairplot_parziale/p2.png'))
axes[2].imshow(mpimg.imread('pairplot_parziale/p3.png'))
#alling left
for ax in axes:
    ax.set_anchor('W')
# turn off x and y axis
[ax.set_axis_off() for ax in axes.ravel()]

plt.tight_layout()
plt.show()

#matrice di correlazione dopo aver pulito i dati dai valori nulli e dalle feature inutili
sea.set(font_scale=0.8)
cor_mat = df.corr()
h1 = sea.heatmap(cor_mat, annot=True)
h1.set_title("Matrice di correlazione - feature selezionate", size=12)
plt.setp(h1.get_xticklabels(), rotation=35, ha="right",
         rotation_mode="anchor")
plt.show()

#controllo se il dataset è sbilanciato


u, inv = np.unique(df['LeagueIndex'], return_inverse=True)
counts = np.bincount(inv)
fig, ax = plt.subplots()

b =plt.bar(u, counts, width=0.3)
plt.xticks(range(1, 9))
plt.title("Bilanciamento del dataset")

for bar in b:
    height = bar.get_height()
    ax.text(bar.get_x() +bar.get_width()/2., 0.99*height,
            '%.2f' % float(height/sum(u)) + "%", ha='center', va='bottom')

plt.show()


# Ho diviso qui il database per mantenere una certa consistenza durante i test
'''
y = df['LeagueIndex']
train, test = train_test_split(df, stratify=y, test_size=0.20, random_state=0)

train.to_csv(r'dataset\Skillcraft_train.csv')
test.to_csv(r'dataset\Skillcraft_test.csv')

df.to_csv(r'dataset\Skillcraft_basic_preprocess.csv')

'''




