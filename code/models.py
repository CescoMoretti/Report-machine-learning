import collections
from os import name
from matplotlib.colors import Colormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.formats import style
import seaborn as sea
import warnings
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, f1_score, roc_curve, precision_recall_curve, roc_auc_score
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
    report.pop('accuracy')
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



df = pd.read_csv('dataset/SkillCraft_basic_preprocess.csv', index_col=0)
#df = pd.read_csv('dataset/SkillCraft1_Dataset.csv', na_values=["?"], index_col=0)
rank_names=['Bronzo','Argento','Oro', 'Platino', 'Diamante', 'Master','Grand Master', 'Pro']

#Grafico bilanciamento dataset
u, inv = np.unique(df['LeagueIndex'], return_inverse=True)
counts = np.bincount(inv)
plt.bar(u, counts, width=0.3)
plt.xticks(range(1, 9))
plt.show()

y = df['LeagueIndex']
X = df.drop(columns=['LeagueIndex'])

#X = df.drop(columns=['LeagueIndex','Age', 'HoursPerWeek', 'TotalHours'])    # Provo a dargli in pasto il dataset con anche player ID
                                                                             # ma non cambia molto, i modelli che ho scelto fanno una buona selezione 
                                                                             # delle feature grazie alle loro caratteristiche di base dando meno peso
                                                                             # a quelle meno importanti, quindi non penso sia necessario un ulteriore
                                                                             # feature selection nel preprocessing

result = pd.DataFrame()

########################################################################################################################
#                                                       MODEL(S)                                                       #
########################################################################################################################
#Softmax regression
softmax_reg_model = LogisticRegression(multi_class='multinomial', solver='saga', penalty = 'l2', class_weight='balanced', max_iter=200)

#Decision tree classifier
decision_tree = DecisionTreeClassifier(class_weight='balanced')

#Ridge regression arrotondando all'intero più vicino
ridge_cv = RidgeCV(alphas=[ 0.0001, 0.001, 0.1, 1.0], cv=StratifiedKFold(3))
########################################################################################################################
#                                          MODEL SELECTION CON CROSS-VALIDATION                                        #
########################################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

#Softmax Regression
c_range = 10.0**np.arange(-5,1)
parameters = {'penalty': ('l1', 'l2'), 'C': c_range}
model = softmax_reg_model
sr_v1 = GridSearchCV(estimator=model, param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(3)) 
sr_v1.fit(X_train, y_train)

print('\n\nSOFTMAX REGRESSION')
print("Trainig result:")
print('Overall, the best choice for parameter penalty is ', sr_v1.best_params_.get('penalty'))
print('the best value for parameter C is ', sr_v1.best_params_.get('C'))
print('the best choice for parameter class_weight is ', sr_v1.best_params_.get('class_weight'))
print(' since these lead to F1-score = ', sr_v1.best_score_)
print("test result")
pred = sr_v1.predict(X_test)
Test_performance(pred,y_test, rank_names, 'SR v1')

#----------------------------------------------------------------------------------------------------------------------
#Decision tree

model = decision_tree
parameters = {"criterion": ["gini", "entropy"], "max_depth": np.arange(10, 100, 10)}

dt_v1 = GridSearchCV(estimator=model,param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(3)) 
dt_v1.fit(X_train, y_train)
print('\n\nDecision Tree')
print("Trainig result:")
print('il migliore parametro per max_dept è ', dt_v1.best_params_.get('max_depth'))
print("")
print('F1-score = ', dt_v1.best_score_)
print("test result")
pred = dt_v1.predict(X_test)
Test_performance(pred,y_test, rank_names, 'DT v1')
#----------------------------------------------------------------------------------------------------------------------
#Ridge regression

rr_v1 = ridge_cv
rr_v1.fit(X_train, y_train)
score = rr_v1.score(X_test, y_test)

print("\n\nRidge regression")
print("il puntgeggio r2 migliore è: ", score)
print("calcolato con alpha uguale a: ", rr_v1.alpha_)
#print(model_r.coef_)
#f1 score train
pred_train = rr_v1.predict(X_train)
pred_train_r = to_next_int(pred_train)
score = f1_score(pred_train_r, y_train, average='weighted')
print("f1 score weighted di train: ", score)

#f1 score test
pred = rr_v1.predict(X_test)
round_pred = to_next_int(pred)
Test_performance(round_pred, y_test, rank_names, 'RR v1')


########################################################################################################################
#                                          PREPROCESSING DEI DATI E V2                                                 #
########################################################################################################################
#Scaling del modello per ridurre la differenza di range tra le feature
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train) 
x_test_scaled = scaler.transform(X_test)

#Softmax Regression con preprocessing
c_range = 10.0**np.arange(-5,1)
parameters = { 'C': c_range}  #'penalty': ['l2'],
model = softmax_reg_model

sr_v2 = GridSearchCV(estimator=model, param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(3))
sr_v2.fit(x_train_scaled, y_train)

print('\n\nSoftmax Regression con preprocessing')
print("trainig result:")
print('Overall, the best choice for parameter penalty is ', sr_v2.best_params_.get('penalty'))
print('the best value for parameter C is ', sr_v2.best_params_.get('C'))
print('the best choice for parameter class_weight is ', sr_v2.best_params_.get('class_weight'))
print(' since these lead to F1-score = ', sr_v2.best_score_)
print("test result")
pred = sr_v2.predict(x_test_scaled)
Test_performance(pred,y_test, rank_names, 'SR v2')

#----------------------------------------------------------------------------------------------------------------------
#Decision tree con preprocessing
model = decision_tree
parameters = {"criterion": ["gini", "entropy"], "max_depth": np.arange(10, 100, 10)}

dt_v2 = GridSearchCV(estimator=model,param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(3)) 
dt_v2.fit(x_train_scaled, y_train)
print('\n\nDecision tree con preprocessing')
print("Trainig result:")
print('il migliore parametro per max_dept è ', dt_v2.best_params_.get('max_depth'))
print('il migliore criterion è ', dt_v2.best_params_.get('criterion'))
print('F1-score = ', dt_v2.best_score_)
print("test result")
pred = dt_v2.predict(x_test_scaled)
Test_performance(pred,y_test, rank_names, 'DT v2')

#----------------------------------------------------------------------------------------------------------------------
#Ridge regression con processing
rr_v2 = ridge_cv
rr_v2.fit(x_train_scaled, y_train)
score = rr_v2.score(x_test_scaled, y_test)


print("\n\nRidge regression con preprocessing")
print("il puntgeggio r2 migliore è: ", score)
print("alpha usato: ", rr_v2.alpha_)
#print(model_r.coef_)

pred_train = rr_v2.predict(x_train_scaled)
pred_train_r = to_next_int(pred_train)
score = f1_score(pred_train_r, y_train, average='weighted')
print("f1 score weighted di train: ", score)

pred = rr_v2.predict(x_test_scaled)
round_pred = to_next_int(pred)
Test_performance(round_pred, y_test, rank_names, 'RR v2')

########################################################################################################################
#                                          ENSEMBLIG DEI MODELLI --> V3                                                #
########################################################################################################################
#Softmax con ensemble e preprocessing

model = AdaBoostClassifier( base_estimator = sr_v2.best_estimator_,
                            n_estimators = 50,
                            random_state = 0
                          )
parameters = {"learning_rate": np.arange(2, 3, 1)} 

sr_v3 = GridSearchCV(estimator=model,param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(5))
sr_v3.fit(x_train_scaled, y_train)
print('\n\nSoftmax Regression con preprocessing in ensamble')
print("Trainig result:")
print('il migliore parametro per learning_rate è ', sr_v3.best_params_.get('learning_rate'))
print('F1-score = ', sr_v3.best_score_)

pred = sr_v3.predict(x_test_scaled)
Test_performance(pred,y_test, rank_names, "SR v3")

#----------------------------------------------------------------------------------------------------------------------

#Decision tree con ensemble senza preprocessing

model = RandomForestClassifier(bootstrap = True)
parameters = {"n_estimators": np.arange(100, 300, 100), "max_depth": np.arange(20, 100, 10)}

dt_v3 = GridSearchCV(estimator=model,param_grid=parameters, scoring='f1_weighted', cv=StratifiedKFold(5)) 
dt_v3.fit(X_train, y_train)
print('\n\ndecision tree con preprocessing e ensamble')
print("Trainig result:")
print('il migliore parametro per max_dept è ', dt_v3.best_params_.get('max_depth'))
print('il migliore parametro per n_estimators è ', dt_v3.best_params_.get('n_estimators'))
print('F1-score = ', dt_v3.best_score_)
print("\ntest result")
pred = dt_v3.predict(X_test)
Test_performance(pred,y_test, rank_names, 'DT v3')

#----------------------------------------------------------------------------------------------------------------------
#Ridge regression con ensemble e preprocessing

model = AdaBoostRegressor(ridge_cv,
                         n_estimators=100,
                         random_state = 0)
parameters = {"learning_rate": np.arange(1, 3, 1), "loss": ["linear", "square","exponential" ]} 

rr_v3 = GridSearchCV(estimator=model, param_grid=parameters, cv=StratifiedKFold(5))
rr_v3.fit(x_train_scaled, y_train)
score = rr_v3.score(x_test_scaled, y_test)


print("\n\nRidge regression boosted")
print("il puntgeggio r2 migliore è: ", score)
print("loss usato: ", rr_v3.best_params_.get('loss'))
print("learning_rate usato: ", rr_v3.best_params_.get('learning_rate'))
pred_train = rr_v3.predict(x_train_scaled)
pred_train_r = to_next_int(pred_train)
score = f1_score(pred_train_r, y_train, average='weighted')
print("f1 score weighted di train: ", score)


pred_test = rr_v3.predict(x_test_scaled)
pred_test_r = to_next_int(pred_test)
Test_performance(pred_test_r, y_test, rank_names, 'RR v3')


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
                          colormap=colors,
                          #alpha= 0.9, 
                          legend= True,
                          xlabel= "",                          
                          ylim =[0, 1],  
                          style="seaborn-deep")
plt.show()
