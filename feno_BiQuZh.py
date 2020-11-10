# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:03:32 2020

@author: Alain
"""

import pandas as pd; 
import numpy as np

df = pd.read_excel('feno_subjects.xlsx') 
x = np.array(df)
print(df)