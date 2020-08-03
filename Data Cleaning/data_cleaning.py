# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 01:46:30 2020

@author: cumea
"""

import pandas as pd

df = pd.read_csv('salary_data_collected.csv')

df.columns

df = pd.read_csv('salary_data_cleaned.csv')

df['Salary Estimate'].describe()


## List Cleaning Objectives by Block

#-remove records with null values for salary field
#-salary parsing
#-company name text only
#-state field and city field
# Job located at Headquarters
#-age of company
#-parsing of job description (pythin, etc.)

## Cleaning Objectives by Block

#-remove records with null values for salary field

df_present_salaries_only = df[df['Salary Estimate'] != -1] #Uneeded

#-salary parsing

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_kD = salary.apply(lambda x: x.replace('K','').replace('$','')) 
df['min_salary'] = minus_kD.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = minus_kD.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.max_salary+df.min_salary)/2


#-company name text only

df['Company Name'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1 )

#-state field and city field

df['job_city'] = df['Location'].apply(lambda x: x.split(', ')[0])
df['job_state'] = df['Location'].apply(lambda x: x[-2:])

# Job located at Headquarters

df['same_location'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

#-age of company

df['age_company'] = df['Founded'].apply(lambda x: x if x<0 else 2020-x)
 

#-parsing of job description (python, etc.)

#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1  if 'python' in x.lower() else 0 )
df.python_yn.value_counts()

#R
df['R_yn'] = df['Job Description'].apply(lambda x: 1  if 'r-studio' or 'r studio' or 'rstudio' in x.lower() else 0 )
df.R_yn.value_counts()

#Tableau
df['tableau_yn'] = df['Job Description'].apply(lambda x: 1  if 'tableau' in x.lower() else 0 )
df.tableau_yn.value_counts()

#Excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1  if 'excel' in x.lower() else 0 )
df.excel_yn.value_counts()

#SAS
df['SAS_yn'] = df['Job Description'].apply(lambda x: 1  if 'sas' in x.lower() else 0 )
df.SAS_yn.value_counts()

# Export Dataframe
df.to_csv('salary_data_cleaned.csv', index = False)

