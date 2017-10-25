import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
from lib2to3.main import main
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sys
from lib2to3.main import main
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score,precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import json

def createFeatureNews(df):

    popular = df[' shares'] >= 1400
    unpopular = df[' shares'] < 1400
    df.loc[popular, 'sharesClass'] = 'popular'
    df.loc[unpopular, 'sharesClass'] = 'unpopular'


def createFeatureSum(df):

    return

def createFeatureNoisySum(df):

    return


def createFeatureSkin(df):
    skin = df['T'] == 1
    noskin = df['T'] == 2

    df.loc[skin, 'TC'] = 1
    df.loc[noskin,'TC'] = 2

chunkSizes = [100,500,1000,5000,10000,50000, 100000, 500000, 1000000,5000000,10000000]



configDict = json.load(open('config.json'))



for c in configDict:

    df = pd.read_csv(c, header=0, sep=configDict[c][1])
    locals()[configDict[c][3]](df)
    features = list(df.columns.values)
    print (c)
    print (df.size)
    print (features)
    #df = df[features[-3:-1]]
    #df = df[0:1000000]


    for s in chunkSizes:
        if (df.size < s):
            print ("NA")
            continue
        chunkedDf = df[0:s]
        regTarget = chunkedDf[configDict[c][0]]
        classTarget = chunkedDf[configDict[c][4]]
        trainingFeatures = chunkedDf.drop([configDict[c][0]],axis=1)
        trainingFeatures = trainingFeatures.drop(configDict[c][2], axis = 1)
        trainingFeatures = trainingFeatures.drop(configDict[c][4], axis=1)

        linReg = linear_model.LinearRegression()
        ridgeReg = linear_model.Ridge(alpha=.5)
        logReg  =LogisticRegression()
        randomForest = RandomForestClassifier(n_estimators=100, n_jobs=-1)

        regTrainFeatures, regTestFeatures, regTrainTarget, regTestTarget = train_test_split(trainingFeatures, regTarget, test_size=0.3,
                                                            random_state=0)

        classTrainFeatures, classTestFeatures, classTrainTarget, classTestTarget = train_test_split(trainingFeatures, classTarget, test_size=0.3,
                                                            random_state=0)


        linReg.fit(regTrainFeatures, regTrainTarget)
        linRegPredict = linReg.predict(regTestFeatures)

        ridgeReg.fit(regTrainFeatures, regTrainTarget)
        ridgeRegPredict = ridgeReg.predict(regTestFeatures)



        print ("chunk size ", s)
        print("linReg:"
              , mean_squared_error(regTestTarget, linRegPredict), " ", r2_score(regTestTarget, linRegPredict))

        print("ridgeReg:"
              , mean_squared_error(regTestTarget, ridgeRegPredict), " ", r2_score(regTestTarget, ridgeRegPredict))


        try:
            randomForest.fit(classTrainFeatures, classTrainTarget)
            randomForestPredict = randomForest.predict(classTestFeatures)
            print("randomForest"
                  , precision_score(classTestTarget, randomForestPredict, average='weighted'), " ", f1_score(classTestTarget, randomForestPredict, average='weighted'))
        except Exception as e:
            print (e)


        try:
            logReg.fit(classTrainFeatures, classTrainTarget)
            logRegPredict = logReg.predict(classTestFeatures)
            print("logReg:"
                  , precision_score(classTestTarget, logRegPredict, average='weighted'), " ", f1_score(classTestTarget, logRegPredict, average='weighted'))
        except Exception as e:
            print (e)


    #    linRegScores = cross_val_score(linReg, trainingFeatures, regTarget, cv=10, scoring='r2')
     #   ridgeScores = cross_val_score(ridgeReg, trainingFeatures, regTarget, cv=10, scoring='r2')

        #logRegScores = cross_val_score(logReg, trainingFeatures, classTarget, cv=10)
        #print "linReg scores ", linRegScores.mean(), " ridgeScores", ridgeScores.mean(), " logRegScores", logRegScores.mean()