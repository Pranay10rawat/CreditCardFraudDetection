# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:13:33 2019

@author: Pranay Rawat
"""
import sys
import numpy 
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python : {} '.format(sys.version))
print('Numpy : {} '.format(numpy.__version__))
print('Pandas : {} '.format(pandas.__version__))
print('Matplotlib :{} '.format(matplotlib.__version__))
print('Seaborn : {} '.format(seaborn.__version__))
print('Scipy : {}'.format(scipy.__version__))
print('sklearn :{}'.format(sklearn.__version__))


#import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('creditcard.csv')

#large dataset making it small
data = data.sample(frac=0.1 , random_state=1)
print(data.shape)

#plot histogram of each parameter
data.hist(figsize = (40,40))
plt.show()

#determine number of fraud cases in dataset
Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]

outlier_fraction = len(Fraud)/ float(len(Valid))
print(outlier_fraction)


print('Fraud Cases : {}'.format(len(Fraud)))
print('Valid Cases : {}'.format(len(Valid)))

#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax=.8,square=True)
plt.show()

#Get all the columns from the Dataframe
columns = data.columns.tolist()

#filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

#store the variable we';; be predicting
target = "Class"

X=data[columns]
Y=data[target]

#print the shapes of X and Y
print(X.shape)
print(Y.shape)


from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define random state 
state = 1

#define the outlier detection methods

classifier = {
         "Isolation Forest" : IsolationForest(max_samples = len(X),
                                              contamination = outlier_fraction,
                                              random_state = state),
         "Local Outlier Factor":LocalOutlierFactor(
                 n_neighbors = 20,
                 contamination = outlier_fraction)
        }

#Fit the model
n_outliers = len(Fraud)

for i , (clf_name , clf) in enumerate (classifier.items()):
    
    #fit the data and tag ouliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred= clf.predict(X)
    
    #Reshape the prediction values to 0 for valid ,1 for fraud
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    
    
    n_errors = (y_pred!=Y).sum()
    
    #Run classification metrics
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
