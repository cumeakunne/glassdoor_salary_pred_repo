# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 00:59:57 2020

@author: cumea
"""

import requests 
from data_input import data_in

url ='http://127.0.0.1:5000/predict'
header = {"Content-Type": "application/json"} 
data =  {"input": data_in}

r = requests.get(url = url, params = header, json=data) 

r.json()