# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 03:11:39 2020

@author: cumea
"""

#import scraper module
import glassdoor_scraper as gs

#define path on local machine
path = "C:/Users/cumea/Desktop/Projects/Glassdoor Salary Prediction/Glassdoor_Salary_Prediction/chromedriver"

#run scraper and store data into dataframe 'df
df = gs.get_jobs('Business Analyst', 500, False, path, 15)

#export data to .csv file
df.to_csv('glassdoor_jobs.csv', index=False)