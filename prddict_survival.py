# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 10:37:20 2020

@author: sambit mohapatra
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("train.csv")

df = data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

#checking for null values


df.dropna(subset=['Embarked'],inplace=True)

#df["Age"] = df["Age"].replace(np.NaN, df["Age"].mean())

df['Age']= df["Age"].interpolate(method='linear', limit_direction='forward', axis=0)

print(df.isnull().sum())

#freq distribution of continuous values vs exited

sns.distplot(df.Age[df.Survived==0])
sns.distplot(df.Age[df.Survived==1])
plt.legend(['Not-Survived','Survived'])
plt.show()

sns.distplot(df.SibSp[df.Survived==0])
sns.distplot(df.SibSp[df.Survived==1])
plt.legend(['Not-Survived','Survived'])
plt.show()

sns.distplot(df.Fare[df.Survived==0])
sns.distplot(df.Fare[df.Survived==1])
plt.legend(['Not-Survived','Survived'])
plt.show()

sns.countplot(df.Sex)
plt.show()

sns.countplot(df.Sex[df.Survived==1])
plt.show()

sns.countplot(df.Parch)
plt.show()

sns.countplot(df.Parch[df.Survived==1])
plt.show()


sns.countplot(df.Pclass)
plt.show()

sns.countplot(df.Pclass[df.Survived==1])
plt.show()

sns.countplot(df.Embarked)
plt.show()

sns.countplot(df.Embarked[df.Survived==1])
plt.show()

corr = df.corr()

plt.figure(figsize=(10,10))
sns.heatmap(corr,cmap='rainbow',annot=True)
plt.show()


ip = df.drop(['SibSp','Survived'],axis=1)
op = df.Survived

#OneHotEncoding the categorical/string columns
 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = 
                       (['Pclass',OneHotEncoder(),[0]],
                        ['Sex',OneHotEncoder(),[1]],
                        ['Embarked',OneHotEncoder(),[5]]),
                       remainder='passthrough')

ip = ct.fit_transform(ip)


from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts = train_test_split(ip,op, test_size=0.2)

#scaling the data = standerization


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xtr)
xtr = sc.transform(xtr)
xts = sc.transform(xts)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(xtr,ytr)
print(model.score(xts,yts))

#confusion matrix


'''
[[ true positive  false positive]
[ false negative  true negative]]
'''


from sklearn.metrics import confusion_matrix

y_pred = model.predict(xts)

print(confusion_matrix(yts, y_pred))

