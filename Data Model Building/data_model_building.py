# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:48:04 2020

@author: cumea
"""

# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read in post eda data
df = pd.read_csv('salary_data_explored.csv')

""" last feature edits """

#drop extra index column
df.drop('Unnamed: 0', inplace=True, axis =1)

#drop 'intern' level from seniority factor 
df=df[df['seniority'] != 'Intern'] 
df.seniority.value_counts() #check Intern level is dropped from dataframe


""" build model """

#select features for model
df.columns

df_model = df[['avg_salary','Size','Company Name','Type of ownership', 'Industry', 
               'Revenue', 'num_comp','job_city','same_location','age_company', 
               'tableau_yn', 'excel_yn','python_yn', 'SAS_yn', 'job_simp', 'seniority']]

#create dummy variables
df_dum = pd.get_dummies(df_model)

#partition data into training and test sets
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

""" Model 1: Mulitple Regression """

# Statsmodel version   
# link=https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLS.html
import statsmodels.api as sm

X_sm = sm.add_constant(X_train)

#make predictions using the training set
model = sm.OLS(y_train, X_sm)
model.fit().summary() # R-squared = 0.652 (e.g. model explains about 65% of the variance in salary estimate)

## SciKit version 
#link=https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#link=https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#link=https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

lm = LinearRegression()
lm.fit(X_train,y_train).score(X_train,y_train)

#make predictions using the training set
y_pred = lm.predict(X_train)

# Output the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_train, y_pred)) #Score = 79.46 | # average prediction is off by about $79,000
# Output the coefficient of determination
print('Coefficient of determination: %.2f' % r2_score(y_train, y_pred)) #R2 = 0.65 | Model accounts for 65% of the variance in data

# The mean squared error via cross validation
np.mean(cross_val_score(lm, X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) #Score = -12127940362.337252 | score is too large for reasonable interpretation



""" Model 2: Lasso Regression """

lm_l = Lasso()
lm_l.fit(X_train,y_train)
#1st Take on Training Set
np.mean(cross_val_score(lm_l, X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) #Default alpha = 1 | #Score = -12.37 | #Prediciton off by roughly $12,000 on average
#1st Take on Test Set
np.mean(cross_val_score(lm_l, X_test,y_test,scoring='neg_mean_absolute_error',cv=3)) #Default alpha = 1 | #Score = -12.27 | #Prediciton off by roughly $12,000 on average

## Tune alpha penalty parameter

#loop through alpha options to see lowest mae 
alpha = []
mae=[]

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=i/100)
    mae.append(np.mean(cross_val_score(lml, X_test,y_test,scoring='neg_mean_absolute_error',cv=3)) )

#plot alpha and mae distribution
plt.plot(alpha,mae)


#determine min absolute mae for best alpha 
err = tuple(zip(alpha,mae))
df_alpha = pd.DataFrame(err,columns=['alpha','mae'])
print(df_alpha[df_alpha['mae'] == max(df_alpha.mae)]) #Best alpha = 0.69 | Best mae = -11.93


""" Model 3: Random Forest Regressor"""
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
np.mean(cross_val_score(rf, X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) #Score = -13.59 | #Prediciton off by roughly $13,000 on average

#tune random forest model with GridseachCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto', 'sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_ #score = -12.59 
gs.best_estimator_ #below are the optimized parameters for our random forest model

""" RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=None,
                      max_features='log2', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=80,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)"""


""" Test Each Model """
ypred_lm = lm.predict(X_test)
ypred_lm1 = lm_l.predict(X_test)
ypred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

#mae score for each model
mean_absolute_error(y_test,ypred_lm) #score = 190543068505.2743 | Prediction error to large to be reasoably explained
mean_absolute_error(y_test,ypred_lm1) #score = 11.53 | Prediction off by roughly $11,500 on average
mean_absolute_error(y_test,ypred_rf) #score = 11.52 | Prediction off by roughly $11,500 on average

#see first 10 predictions
y_test[:10].round(0) 
ypred_lm[:10].round(0) 
ypred_lm1[:10].round(0) 
ypred_rf[:10].round(0) 


""" Test Ensembles """

#loop through best ensemble pair weights
weight1=[]
mae =[]

for i in range(1,10):
    weight1.append(i/10)
    mae.append(mean_absolute_error(y_test,(ypred_rf*(i/10)+ypred_lm1*(1-i/10))))

#plot ensemble pair weights
plt.plot(weight1,mae)

#determine min absolute mae for best alpha 
w = tuple(zip(weight1,mae))
w_df = pd.DataFrame(w,columns=['weight','mae'])

w_df[w_df['mae']==min(w_df['mae'])] #Best Weight = 0.5 | score = 11.29 Best MAE* 



""" Model Deployment """

#for simplicity, we use the random forest for deployment as the model performance is comparable to ensemble
import pickle
pickl = {'model':gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )


file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(X_test.iloc[1,:].values.reshape(1,-1)).round(1)*1000 #predicted salary estimate = $61,900

##step used to create the data input list to test Flask API
#list(X_test.iloc[1,:])