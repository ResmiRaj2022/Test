import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
trai = pd.read_csv("Train.csv")
print(trai.head())
#iris.drop("Id", axis=1, inplace = True)
y = trai['cost_category']
#iris.drop(columns='Species',inplace=True)
#X = iris[[ 'country', 'age_group', 'tour_arrangement']]
X = trai[['country', 'age_group', 'tour_arrangement','purpose']]
# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)

pickle.dump(model,open('model.pkl', 'wb'))

