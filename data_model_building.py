# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:48:04 2020

@author: cumea
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read in post eda data
df = pd.read_csv('salary_data_explored.csv')

#drop extra index column
df.drop('Unnamed: 0', inplace=True, axis =1)

"""List Tasks in Model Building Step"""



#-choose relevant columns/features
df.columns

df=df[df['seniority'] != 'Intern'] #drop Intern level from dataframe
df.seniority.value_counts() #check Intern level is dropped from dataframe

df = df[df != '-1']

df_model_kj = df[['avg_salary', 'Rating', 'Size','Type of ownership', 'Industry', 'Sector', 'Revenue', 
               'num_comp', 'desc_len','job_city','same_location','age_company','python_yn',
               'tableau_yn', 'excel_yn', 'SAS_yn', 'job_simp', 'seniority']] #missing employer provided and hourly



df_model = df[['avg_salary','Size','Company Name','Type of ownership', 'Industry',
               'Revenue', 'num_comp','job_city','same_location','age_company', 
               'tableau_yn', 'excel_yn', 'job_simp', 'seniority']] #Drop'python_yn', , 'SAS_yn', desc_len, 'Rating', 'Sector', Add Company Name

#-create dummy variables
df_dum = pd.get_dummies(df_model)

#-train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

#-m1: mulitple regression

## Statsmodel version   link=https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html
import statsmodels.api as sm

X_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_sm)
model.fit().summary()

# R-squared = 0.641 (e.g. model explains about 64% of the variance in salary price)

## SciKit version 
#link=https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#link=https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#link=https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

lm = LinearRegression()
lm.fit(X_train,y_train).score(X_train,y_train)

# Make predictions using the testing set
y_pred = lm.predict(X_train)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_train, y_pred)) #Score=80.48 | #Prediction off by over $80,000
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(y_train, y_pred)) #R2 =0.64 | Model accounts for 64% of the variance in data

# The mean squared error via cross validation
np.mean(cross_val_score(lm, X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) #Score is too large for reasonable interpretation

#-m2: lasso regression
lm_l = Lasso(alpha=0.5)
lm_l.fit(X_train,y_train)
#1st Take on Training Set
np.mean(cross_val_score(lm_l, X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) #Default alpha = 1 | #Score = -12.36 | #Prediciton off by roughly $12,000
#1st Take on Test Set
np.mean(cross_val_score(lm_l, X_test,y_test,scoring='neg_mean_absolute_error',cv=3)) #Default alpha = 1 | #Score = -12.27 | #Prediciton off by roughly $12,000

##tune alpha penalty parameter
#loop through alpha options and plot distribution

alpha = []
mae=[]

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=i/100)
    mae.append(np.mean(cross_val_score(lml, X_test,y_test,scoring='neg_mean_absolute_error',cv=3)) )

plt.plot(alpha,mae)


#determine max mae for best alpha 
err = tuple(zip(alpha,mae))
df_alpha = pd.DataFrame(err,columns=['alpha','mae'])
print(df_alpha[df_alpha['mae'] == max(df_alpha.mae)]) #Best alpha = 0.5 | mae = -11.85

#-m3: random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) #Default rf| #Score = -13.45 | #Prediciton off by roughly $13,000

#-tune models with GridseachCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto', 'sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_ #score = -12.65 
gs.best_estimator_ 

#-test ensembles
ypred_lm = lm.predict(X_test)
ypred_lm1 = lm_l.predict(X_test)
ypred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,ypred_lm) #score = 247630661127.73917
mean_absolute_error(y_test,ypred_lm1) #score = 11.47 | off by roughly $11,000
mean_absolute_error(y_test,ypred_rf) #score = 11.30 | Off by roughly $11,000

r2_score(y_test, ypred_lm) 

y_test[:10].round(0) 
ypred_lm1[:10].round(0) 

# loop through best ensemble

weight1=[]
mae =[]

for i in range(1,10):
    weight1.append(i/10)
    mae.append(mean_absolute_error(y_test,(ypred_rf*(i/10)+ypred_lm1*(1-i/10))))

plt.plot(weight1,mae)

w = tuple(zip(weight1,mae))
w_df = pd.DataFrame(w,columns=['weight','mae'])

w_df[w_df['mae']==min(w_df['mae'])] #Best Weight = 0.5 | score = 11.10 Best MAE* 

r2_score(y_test,(ypred_rf*0.5+ypred_lm1*0.5))

# Model Deployment
#model1 = lm_l 
#model2 = gs.best_estimator_

import pickle
pickl = {'model':gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )


file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(X_test.iloc[1,:].values.reshape(1,-1)).round(1)*1000 #59600


np.array(list(X_test.iloc[1,:])).reshape(1,-1)

list(X_test.iloc[1,:])