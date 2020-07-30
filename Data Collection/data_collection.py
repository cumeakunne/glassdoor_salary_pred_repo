# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 03:11:39 2020

@author: cumea
"""

import glassdoor_scraper as gs
import pandas as pd

path = "C:/Users/cumea/Desktop/Projects/Glassdoor Salary Prediction/Glassdoor_Salary_Prediction/chromedriver"

#df = gs.get_jobs('Business Analyst', 500, False, path, 15)

df.to_csv('glassdoor_jobs.csv', index=False)