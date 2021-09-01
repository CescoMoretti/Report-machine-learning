import collections
from os import name
from matplotlib.colors import Colormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.formats import style
import seaborn as sea
import time
import warnings

from sklearn.metrics import classification_report, f1_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import  RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

def Test_performance(pred,y_test, rank_names, name): #Funzione per valutare le performance dei modelli in base alle varie classi 
    global result
    global names
    global f1_score_weighted
    report = classification_report(y_test, pred, target_names=rank_names, output_dict=True)
    report.pop('accuracy') #il dataset è sbilanciato, non è una metrica accurata per questo caso
    print ("{:<15} {:<20} {:<20} {:<20} {:<20}".format('Rank','Precision','Recall','F1 score','Support'))
    for k, v in report.items():    
        print ("{:<15} {:<20} {:<20} {:<20}  {:<20}".format(k, v['precision'],v['recall'],v['f1-score'],v['support']))
    
    #memorizzo i risultati per la valutazione
    dfr = pd.DataFrame(report)
    dfr.to_excel('risultati/'+name+'.xlsx')    
    df2 = pd.DataFrame({"name":[name],
                        "f1 weighted": [report['weighted avg']['f1-score']],
                        "f1 macro": [report['macro avg']['f1-score']],
                        "precision weighted":[report['weighted avg']['precision']],
                        "precision macro": [report['macro avg']['precision']],
                        "recall weighted":[report['weighted avg']['recall']],
                        "recall macro": [report['macro avg']['recall']],                       
                        })   
    result = result.append(df2, ignore_index = True)
 

def to_next_int(pred): #Funzione per arrotondare il risultato della ridge regression alle classi target
    round_pred = np.rint(pred)
    round_pred = np.clip(round_pred, 1, 8)
    return round_pred

def measure_time(start, nome, tipo):
    global tempi
    end = time.time()
    time_tot = end - start
    dft = pd.DataFrame({'modello': [nome],
                        'tipo': [tipo],
                        'tempo': [time_tot]})
    tempi = tempi.append(dft)
    return time_tot






df_tot = pd.read_csv('dataset/SkillCraft_basic_preprocess.csv', index_col=0)
rank_names=['Bronzo','Argento','Oro', 'Platino', 'Diamante', 'Master','Grand Master', 'Pro']

'''
#Grafico bilanciamento dataset
u, inv = np.unique(df_tot['LeagueIndex'], return_inverse=True)
counts = np.bincount(inv)
plt.bar(u, counts, width=0.3)
plt.xticks(range(1, 9))
plt.show()'''

df = pd.read_csv('dataset/SkillCraft_train.csv', index_col=0)
y_train = df['LeagueIndex']
X_train = df.drop(columns=['LeagueIndex'])
df = pd.read_csv('dataset/SkillCraft_test.csv', index_col=0)
y_test = df['LeagueIndex']
X_test = df.drop(columns=['LeagueIndex'])


result = pd.DataFrame()
tempi = pd.DataFrame()

########################################################################################################################
#                                                       MODEL(S)                                                       #
########################################################################################################################
#Softmax regression
softmax_reg_model = LogisticRegression(multi_class='multinomial', solver='saga', penalty = 'l2', class_weight='balanced', max_iter=200)

#Decision tree classifier
decision_tree = DecisionTreeClassifier(class_weight='balanced')

#Ridge regression arrotondando all'intero più vicino
ridge_cv = RidgeCV(alphas=[ 0.0001, 0.001, 0.1, 1.0, 10, 100], cv=StratifiedKFold(5))
########################################################################################################################
#                                          MODEL SELECTION CON CROSS-VALIDATION                                        #
########################################################################################################################


#Softmax Regression
nome = 'SR v1'
c_range = 10.0**np.arange(-5,1)
parameters = {'C': c_range}
model = softmax_reg_model

sr_v1 = GridSearchCV(estimator=model, param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(5)) 
start = time.time()
sr_v1.fit(X_train, y_train)
t_train = measure_time(start, nome, "train")


print('\n\nSOFTMAX REGRESSION')
print("tempo addestramento: ", t_train)
print("Trainig result:")
print('Overall, the best choice for parameter penalty is ', sr_v1.best_params_.get('penalty'))
print('the best value for parameter C is ', sr_v1.best_params_.get('C'))
print('the best choice for parameter class_weight is ', sr_v1.best_params_.get('class_weight'))
print(' since these lead to F1-score = ', sr_v1.best_score_)
print("test result")

start = time.time()
pred = sr_v1.predict(X_test)
t_test = measure_time(start, nome, "test")
Test_performance(pred,y_test, rank_names, nome)


#----------------------------------------------------------------------------------------------------------------------
#Decision tree
nome = 'DT v1'
model = decision_tree
parameters = {"criterion": ["gini", "entropy"], "max_depth": np.arange(10, 100, 10)}

dt_v1 = GridSearchCV(estimator=model,param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(5)) 
start = time.time()
dt_v1.fit(X_train, y_train)
t_train = measure_time(start, nome, "train")
print('\n\nDecision Tree')
print("tempo addestramento: ", t_train)
print("Trainig result:")
print('il migliore parametro per max_dept è ', dt_v1.best_params_.get('max_depth'))
print("")
print('F1-score = ', dt_v1.best_score_)
print("test result")
start = time.time()
pred = dt_v1.predict(X_test)
t_test = measure_time(start, nome, "test")
Test_performance(pred,y_test, rank_names, nome)
#----------------------------------------------------------------------------------------------------------------------
#Ridge regression
nome = 'RR v1'
rr_v1 = ridge_cv
start = time.time()
rr_v1.fit(X_train, y_train)
t_train = measure_time(start, nome, "train")

start = time.time()
pred = rr_v1.predict(X_test)
t_test = measure_time(start, nome, "test")
mse = mean_squared_error(pred, y_test)

print("\n\nRidge regression")
print("tempo addestramento: ", t_train)
print("l'mse è: ", mse)
print("calcolato con alpha uguale a: ", rr_v1.alpha_)
#print(model_r.coef_)
#f1 score train
pred_train = rr_v1.predict(X_train)
pred_train_r = to_next_int(pred_train)
score = f1_score(pred_train_r, y_train, average='weighted')
print("f1 score weighted di train: ", score)

#f1 score test
round_pred = to_next_int(pred)
Test_performance(round_pred, y_test, rank_names, nome)


########################################################################################################################
#                                          PREPROCESSING DEI DATI E V2                                                 #
########################################################################################################################

#Feature selection, inizialmente ho solo tolto le feature che ritenevo inutili, poi se verifico la presenza di
# multi collinearità farò un ulteriore feature selection
# PS: non penso che ci siano problemi di multicollinearità, gli altri algoritmi in ensemble hanno circa gli stessi
# risultati rispetto a random forest che non soffre a causa della multicollinearità
x_train_FS = X_train.drop(columns=['GameID'])
x_test_FS = X_test.drop(columns=['GameID'])
#Scaling del modello per ridurre la differenza di range tra le feature
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train_FS) 
x_test_scaled = scaler.transform(x_test_FS)

#Softmax Regression con preprocessing
nome = 'SR v2'
c_range = 10.0**np.arange(-5,1)
parameters = { 'C': c_range}  # 'penalty': ['l1','l2'], ho dovuto rimuovere la penalty perchè se viene selezionata a l1 l'ensemble diventa pessimo
model = softmax_reg_model

sr_v2 = GridSearchCV(estimator=model, param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(5))
start = time.time()
sr_v2.fit(x_train_scaled, y_train)
t_train = measure_time(start, nome, "train")

print('\n\nSoftmax Regression con preprocessing')
print("tempo addestramento: ", t_train)
print("trainig result:")
print('Overall, the best choice for parameter penalty is ', sr_v2.best_params_.get('penalty'))
print('the best value for parameter C is ', sr_v2.best_params_.get('C'))
print('the best choice for parameter class_weight is ', sr_v2.best_params_.get('class_weight'))
print(' since these lead to F1-score = ', sr_v2.best_score_)
print("test result")

start = time.time()
pred = sr_v2.predict(x_test_scaled)
t_test = measure_time(start, nome, "test")

Test_performance(pred,y_test, rank_names, nome)
#----------------------------------------------------------------------------------------------------------------------
#Decision tree con preprocessing
nome = 'DT v2'
model = decision_tree
parameters = {"criterion": ["gini", "entropy"], "max_depth": np.arange(10, 100, 10)}

dt_v2 = GridSearchCV(estimator=model,param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(5)) 
start = time.time()
dt_v2.fit(x_train_scaled, y_train)
t_train = measure_time(start, nome, "train")

print('\n\nDecision tree con preprocessing')
print("tempo addestramento: ", t_train)
print("Trainig result:")
print('il migliore parametro per max_dept è ', dt_v2.best_params_.get('max_depth'))
print('il migliore criterion è ', dt_v2.best_params_.get('criterion'))
print('F1-score = ', dt_v2.best_score_)
print("test result")

start = time.time()
pred = dt_v2.predict(x_test_scaled)
t_test = measure_time(start, nome, "test")

Test_performance(pred,y_test, rank_names, nome)

#----------------------------------------------------------------------------------------------------------------------
#Ridge regression con processing
nome = 'RR v2'
rr_v2 = ridge_cv
start = time.time()
rr_v2.fit(x_train_scaled, y_train)
t_train = measure_time(start, nome, "train")

start = time.time()
pred = rr_v2.predict(x_test_scaled)
t_test = measure_time(start, nome, "test")
mse = mean_squared_error(pred, y_test)


print("\n\nRidge regression con preprocessing")
print("tempo addestramento: ", t_train)
print("l'mse è: ", mse)
print("alpha usato: ", rr_v2.alpha_)
#print(model_r.coef_)

pred_train = rr_v2.predict(x_train_scaled)
pred_train_r = to_next_int(pred_train)
score = f1_score(pred_train_r, y_train, average='weighted')
print("f1 score weighted di train: ", score)


round_pred = to_next_int(pred)
Test_performance(round_pred, y_test, rank_names, nome)

########################################################################################################################
#                                          ENSEMBLIG DEI MODELLI --> V3                                                #
########################################################################################################################
#Softmax con ensemble e preprocessing
nome =  "SR v3"
model = AdaBoostClassifier( base_estimator = sr_v2.best_estimator_,
                            n_estimators = 50,
                            random_state = 0
                          )
parameters = {"learning_rate": np.arange(2, 3, 1)} 

sr_v3 = GridSearchCV(estimator=model,param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(5))
start = time.time()
sr_v3.fit(x_train_scaled, y_train)
t_train = measure_time(start, nome, "train")

print('\n\nSoftmax Regression con preprocessing in ensamble')
print("tempo addestramento: ", t_train)
print("Trainig result:")
print('il migliore parametro per learning_rate è ', sr_v3.best_params_.get('learning_rate'))
print('F1-score = ', sr_v3.best_score_)

start = time.time()
pred = sr_v3.predict(x_test_scaled)
t_test = measure_time(start, nome, "test")

Test_performance(pred,y_test, rank_names, nome)

#----------------------------------------------------------------------------------------------------------------------

#Decision tree con ensemble senza preprocessing
nome =  'DT v3'
model = RandomForestClassifier(bootstrap = True, criterion=dt_v1.best_params_.get('criterion'))
parameters = {"n_estimators": np.arange(100, 300, 100),  "max_depth": np.arange(10, 100, 10)}

dt_v3 = GridSearchCV(estimator=model,param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(5)) 
start = time.time()
dt_v3.fit(X_train, y_train)
t_train = measure_time(start, nome, "train")

print('\n\ndecision tree con preprocessing e ensamble')
print("tempo addestramento: ", t_train)
print("Trainig result:")
print('il migliore parametro per max_dept è ', dt_v3.best_params_.get('max_depth'))
print('il migliore parametro per n_estimators è ', dt_v3.best_params_.get('n_estimators'))
print('F1-score = ', dt_v3.best_score_)
print("\ntest result")
start = time.time()
pred = dt_v3.predict(X_test)
t_test = measure_time(start, nome, "test")
Test_performance(pred,y_test, rank_names, nome)


#----------------------------------------------------------------------------------------------------------------------
#Ridge regression con ensemble e preprocessing
nome = 'RR v3'
model = AdaBoostRegressor(ridge_cv,
                         n_estimators=50,
                         random_state = 0)
parameters = {"learning_rate": np.arange(1, 3, 1)} 

rr_v3 = GridSearchCV(estimator=model, param_grid=parameters, cv=StratifiedKFold(5))
start = time.time()
rr_v3.fit(x_train_scaled, y_train)
t_train = measure_time(start, nome, "train")

start = time.time()
pred = rr_v3.predict(x_test_scaled)
t_test = measure_time(start, nome, "test")

mse = mean_squared_error(pred, y_test)


print("\n\nRidge regression boosted")
print("tempo addestramento: ", t_train)
print("l'mse è: ", mse)
print("loss usato: ", rr_v3.best_params_.get('loss'))
print("learning_rate usato: ", rr_v3.best_params_.get('learning_rate'))
pred_train = rr_v3.predict(x_train_scaled)
pred_train_r = to_next_int(pred_train)
score = f1_score(pred_train_r, y_train, average='weighted')
print("f1 score weighted di train: ", score)


pred_test_r = to_next_int(pred)

Test_performance(pred_test_r, y_test, rank_names, nome)


########################################################################################################################
#                                                      VALUTAZIONE                                                     #
########################################################################################################################

result.set_index('name',inplace=True)
result = result.transpose()
arg = result.columns.values.tolist()
colors = sea.color_palette("crest", as_cmap=True)
result.reset_index().plot(x='index',                            
                          y=arg,
                          kind="bar",
                          width=0.6,
                          figsize=(10,7), 
                          rot=0,
                          #colormap=colors,
                          #alpha= 0.9, 
                          legend= True,
                          xlabel= "",                          
                          ylim =[0, 1],  
                          style="seaborn-deep")
plt.show()
tempi_train = tempi.loc[tempi['tipo'] == 'train']
t_plot = sea.barplot(x="modello",
                     y="tempo",                     
                     data=tempi_train)
t_plot.set(yscale="log")

#plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.show()