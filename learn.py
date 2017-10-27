#/*********************************************************************/
#   learn.py -- Implementation for Machine Learning Algorithms         *
#               using scikit-learn.                                    *
#     Authors:  Kariem Fahmi                                           *
#               Adwitiya Chakraborty                                   *
#               Salil Ajgoankar                                        *
#                                                                      *
#      Purpose: Evaluate ML Algorithms in Different Real Life datasets.*
#                                                                      *
#               GitHub Repo: https://goo.gl/F9zbHp                     *
#                Build Date: 24.10.2017                                *
#/*********************************************************************/
import numpy as np
import pandas as pd
import json
import os
from sklearn.linear_model \
    import LogisticRegression
from sklearn \
    import linear_model
from sklearn.model_selection \
    import train_test_split
from sklearn.metrics \
    import mean_squared_error, r2_score, f1_score,precision_score
from sklearn.ensemble \
    import RandomForestClassifier

# Class for handling Output Colors
class bcolors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    OKBLUE = '\033[94m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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

# Chunk Sizes for datasets
chunkSizes = [100,500,1000,5000,10000,50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
fileLinesFinal = []

try:

#   Loading The config file
    configDict = json.load(open('config.json'))
except Exception as e:
    print (bcolors.FAIL+"Unable to Load config file --", e)
    quit()

for c in configDict:

    absPathDataset = os.path.abspath(c)
    df = pd.read_csv(absPathDataset, header=0, sep=configDict[c][1])
    locals()[configDict[c][3]](df)
    features = list(df.columns.values)
    print (bcolors.HEADER+bcolors.BOLD+"Data Set:",c)
    print (bcolors.UNDERLINE+"Number of Instances:",df.size)

    for s in chunkSizes:
        if (df.size < s):
            print (bcolors.FAIL+"Cannot Compute more than chunk sizes.")
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


        fileLines = []
        tempLine = "Chunk Size: %s"%s
        fileLines.append(tempLine)
        print (bcolors.BOLD+"Chunk Size:", s)
        print (bcolors.OKBLUE+"Regression Algorithms:")
        print(bcolors.OKGREEN+"Linear Regression --",
              c ,"RMSE: %.15f" %np.sqrt(mean_squared_error(regTestTarget, linRegPredict)), "R2 Score:", r2_score(regTestTarget, linRegPredict))
        tempLine = "Linear Regression -- %s" %c  + " RMSE:%.15f" %np.sqrt(mean_squared_error(regTestTarget, linRegPredict)) + " R2 Score:%f" %r2_score(regTestTarget, linRegPredict)
        fileLines.append(tempLine)

        print(bcolors.OKGREEN+"Ridge Regression --",
              c , "RMSE:",np.sqrt(mean_squared_error(regTestTarget, ridgeRegPredict)), "R2 Score:", r2_score(regTestTarget, ridgeRegPredict))

        tempLine = "Ridge Regression -- %s"%c+" RMSE:%s"%np.sqrt(mean_squared_error(regTestTarget, ridgeRegPredict))+" R2 Score:%s"%r2_score(regTestTarget, ridgeRegPredict)
        fileLines.append(tempLine)

        print(bcolors.OKBLUE + "Classification Algorithms:")

        try:
            randomForest.fit(classTrainFeatures, classTrainTarget)
            randomForestPredict = randomForest.predict(classTestFeatures)
            print(bcolors.OKGREEN+"Random Forest --",
                  c,"Accuracy:", precision_score(classTestTarget, randomForestPredict, average='weighted'), "f1 Score:", f1_score(classTestTarget, randomForestPredict, average='weighted'))
            tempLine = "Random Forest -- %s"%c + " Accuracy:%s" %precision_score(classTestTarget, randomForestPredict, average='weighted')+" f1 Score:%s"%f1_score(classTestTarget, randomForestPredict, average='weighted')
            fileLines.append(tempLine)
        except Exception as e:
            print (e)

        try:
            logReg.fit(classTrainFeatures, classTrainTarget)
            logRegPredict = logReg.predict(classTestFeatures)
            print(bcolors.OKGREEN+"Logistic Regression--",
                  c,"Accuracy:", precision_score(classTestTarget, logRegPredict, average='weighted'), "f1 Score:", f1_score(classTestTarget, logRegPredict, average='weighted'))

            tempLine = "Logistic Regression-- %s" %c + " Accuracy:%s" %precision_score(classTestTarget, logRegPredict, average='weighted') + " f1 Score:%s"%f1_score(classTestTarget, logRegPredict, average='weighted')
            fileLines.append(tempLine)
            fileLinesFinal.extend(fileLines)
        except Exception as e:
            print (e)

try:
    absPathOutput = os.path.abspath("output/output.txt")
    outFile = open(absPathOutput, 'w')
    outFile.write("\n".join(fileLinesFinal))
except Exception as e:
    print(e)