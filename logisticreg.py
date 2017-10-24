import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

input_file = "OnlineNewsPopularity.csv"

df = pd.read_csv(input_file, header=0)

features = list(df.columns.values)
#print (features[1:])
#df = df[features[-3:-1]]
#df = df[0:1000000]
X_train, X_test, y_train, y_test = train_test_split(df[features[1:-1]],df[' shares'],test_size=0.9,random_state=0)
logreg = linear_model.LinearRegression()
logreg.fit(X_train, y_train)

print('Coefficients: \n', logreg.coef_)


pred = logreg.predict(X_test)

print(pred, " ", y_test)
print("Mean squared error:"
      ,mean_squared_error(y_test, pred))





#print (df[features])
#target = 'shares'
#print(df.shares)
#numpy_array = df.as_matrix()

#logreg.fit(numpy_array,df['shares'])
#print(logreg.predict_proba(numpy_array,df['shares']))

#print(numpy_array)
#print(df['url'])


