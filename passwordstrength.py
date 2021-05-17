# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:50:53 2020

@author: Shubhangi sakarkar
"""

import pandas as pd
import numpy as np

data=pd.read_csv('data.csv',',',error_bad_lines=False)

## preproceeing the data

data.isnull().sum()
data.dropna(inplace=True)


password_tuple=np.array(data)
import random
random.shuffle(password_tuple)

## X and y creation
y=[labels[1] for labels in password_tuple]
X=[labels[0] for labels in password_tuple]
## creating features from single word
def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

## feature creations

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=word_divide_char)
X=vectorizer.fit_transform(X)
feature_names = vectorizer.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=X[0]
 

df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)



## applying machine learning model logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
lr=LogisticRegression(penalty='l2',multi_class='ovr')
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)


#print the scores
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)

## Multinomial

clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
clf.fit(X_train, y_train) #training
print(clf.score(X_test, y_test))

## test with new data
X_predict=np.array(["@@##678dfgs"])
X_predict=vectorizer.transform(X_predict)
y_pred1=lr.predict(X_predict)
print(y_pred1)




## apllying XGBoost 
import xgboost as xgb
xgb_classifier=xbg.XGBClassifier()
xgb_classifier.fit(X_train,y_train)