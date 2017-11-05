#Before starting analysis, data preprocessing is required

#Import required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset

dataset = pd.read_csv('Data.csv')

#Define Dependent as y and independent variables as X

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, :3]

#Taking care of missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])




