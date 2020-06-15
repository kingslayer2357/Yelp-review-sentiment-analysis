# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:21:49 2020

@author: kingslayer
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

yelp_df=pd.read_csv(r"yelp.csv")

sns.countplot(x="stars",data=yelp_df)

yelp_df.describe()
yelp_df.info()


yelp_df["length"]=yelp_df["text"].apply(len)

plt.hist(yelp_df["length"])


g=sns.FacetGrid(data=yelp_df,col="stars")
g.map(plt.hist,"length",bins=20,color="r")


#DATA CLEANING

#Removing Punctuations

import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
   



def textcleaning(data):
    data=[char for char in data if char not in string.punctuation]
    data="".join(data)
    data=data.lower()
    data=[word for word in data.split() if word not in stopwords.words("english")]
    data=" ".join(data)
    return data

yelp_df_clean=yelp_df["text"].apply(textcleaning)


cv=CountVectorizer()
X=cv.fit_transform(yelp_df_clean).toarray()
yelp_df['stars'][yelp_df["stars"]==1]=0
yelp_df['stars'][yelp_df["stars"]==2]=0
yelp_df['stars'][yelp_df["stars"]==3]=1
yelp_df['stars'][yelp_df["stars"]==4]=1
yelp_df['stars'][yelp_df["stars"]==5]=1
y=yelp_df["stars"]

sns.countplot(y='stars',data=yelp_df)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Model
from sklearn.naive_bayes import MultinomialNB
nlp=MultinomialNB()
nlp.fit(X_train,y_train)

y_pred=nlp.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

print(classification_report(y_test,y_pred))


#Model -2

from sklearn.linear_model import LogisticRegression
nlp=LogisticRegression()
nlp.fit(X_train,y_train)

y_pred=nlp.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

print(classification_report(y_test,y_pred))

