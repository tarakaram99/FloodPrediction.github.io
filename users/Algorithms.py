
from django.conf import settings
# importing the libraries..
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.tree import DecisionTreeClassifier
import scipy
from keras.callbacks import Callback
from scipy.stats import spearmanr
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,classification_report

filepath = settings.MEDIA_ROOT + "\\" + "rainfall in india 1901-2015.csv"





def logistic():
    # importing the dataset..
    x = pd.read_csv(filepath)
    # checking the null values in the dataset...
    x['JAN'] = x['JAN'].fillna(x['JAN'].mean())
    x['FEB'] = x['FEB'].fillna(x['FEB'].mean())
    x['MAR'] = x['MAR'].fillna(x['MAR'].mean())
    x['APR'] = x['APR'].fillna(x['APR'].mean())
    x['MAY'] = x['MAY'].fillna(x['MAY'].mean())
    x['JUN'] = x['JUN'].fillna(x['JUN'].mean())
    x['JUL'] = x['JUL'].fillna(x['JUL'].mean())
    x['AUG'] = x['AUG'].fillna(x['AUG'].mean())
    x['SEP'] = x['SEP'].fillna(x['SEP'].mean())
    x['OCT'] = x['OCT'].fillna(x['OCT'].mean())
    x['NOV'] = x['NOV'].fillna(x['NOV'].mean())
    x['DEC'] = x['DEC'].fillna(x['DEC'].mean())
    x['ANNUAL'] = x['ANNUAL'].fillna(x['ANNUAL'].mean())
    x['Jan-Feb'] = x['Jan-Feb'].fillna(x['Jan-Feb'].mean())
    x['Mar-May'] = x['Mar-May'].fillna(x['Mar-May'].mean())
    x['Jun-Sep'] = x['Jun-Sep'].fillna(x['Jun-Sep'].mean())
    x['Oct-Dec'] = x['Oct-Dec'].fillna(x['Oct-Dec'].mean())
    y1 = list(x["YEAR"])
    x1 = list(x["Jun-Sep"])
    z1 = list(x["JUN"])
    w1 = list(x["MAY"])
    flood = []
    june = []
    sub = []

    # CREATING A NEW COLOUMN WITH BINARY CLASSIFICATION DEPENDING IF THAT YEAR HAD FLOODED OR NOT, USING RAINFALL OF THAT YEAR AS THRESHOLD
    # print(x1[114])
    for i in range(0, len(x1)):
        if x1[i] > 2400:
            flood.append('1')
        else:
            flood.append('0')

    # print(len(x1))

    # APPROAXIMATELY FINDING THE RAINFALL DATA FOR 10 DAYS FOR THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        june.append(z1[k] / 3)

    # FINDING THE INCREASE IN RAINFALL FROM THE MONTH OF MAY TO THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        sub.append(abs(w1[k] - z1[k]))

    # print(len(flood),len(x1))
    df = pd.DataFrame({'flood': flood})
    df1 = pd.DataFrame({'per_10_days': june})

    x["flood"] = flood
    x["avgjune"] = june
    x["sub"] = sub

    # SAVING THE NEW CSV FILE WITH THE NEW COLOUMNS
    x.to_csv(settings.MEDIA_ROOT + "\\"+"out1.csv")

    df1 = pd.read_csv(settings.MEDIA_ROOT + "\\"+"out1.csv")

    X = df1.iloc[:, [17, 18, 19, 16]].values
    y1 = df1.iloc[:, 20].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y1,test_size=0.3)

    Lr = LogisticRegression()

    Lr.fit(X_train, Y_train)
    pred = Lr.predict(X_test)
    score=Lr.score(X, y1)
    cm = confusion_matrix(pred, Y_test)
    acc = accuracy_score(pred, Y_test)
    fs = f1_score(pred, Y_test)
    return score,fs

def decision():
    # importing the dataset..
    x = pd.read_csv(filepath)

    # checking the null values in the dataset...

    x['JAN'] = x['JAN'].fillna(x['JAN'].mean())
    x['FEB'] = x['FEB'].fillna(x['FEB'].mean())
    x['MAR'] = x['MAR'].fillna(x['MAR'].mean())
    x['APR'] = x['APR'].fillna(x['APR'].mean())
    x['MAY'] = x['MAY'].fillna(x['MAY'].mean())
    x['JUN'] = x['JUN'].fillna(x['JUN'].mean())
    x['JUL'] = x['JUL'].fillna(x['JUL'].mean())
    x['AUG'] = x['AUG'].fillna(x['AUG'].mean())
    x['SEP'] = x['SEP'].fillna(x['SEP'].mean())
    x['OCT'] = x['OCT'].fillna(x['OCT'].mean())
    x['NOV'] = x['NOV'].fillna(x['NOV'].mean())
    x['DEC'] = x['DEC'].fillna(x['DEC'].mean())
    x['ANNUAL'] = x['ANNUAL'].fillna(x['ANNUAL'].mean())
    x['Jan-Feb'] = x['Jan-Feb'].fillna(x['Jan-Feb'].mean())
    x['Mar-May'] = x['Mar-May'].fillna(x['Mar-May'].mean())
    x['Jun-Sep'] = x['Jun-Sep'].fillna(x['Jun-Sep'].mean())
    x['Oct-Dec'] = x['Oct-Dec'].fillna(x['Oct-Dec'].mean())

    y1 = list(x["YEAR"])
    x1 = list(x["Jun-Sep"])
    z1 = list(x["JUN"])
    w1 = list(x["MAY"])



    flood = []
    june = []
    sub = []

    # CREATING A NEW COLOUMN WITH BINARY CLASSIFICATION DEPENDING IF THAT YEAR HAD FLOODED OR NOT, USING RAINFALL OF THAT YEAR AS THRESHOLD
    # print(x1[114])
    for i in range(0, len(x1)):
        if x1[i] > 2400:
            flood.append('1')
        else:
            flood.append('0')

    # print(len(x1))

    # APPROAXIMATELY FINDING THE RAINFALL DATA FOR 10 DAYS FOR THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        june.append(z1[k] / 3)

    # FINDING THE INCREASE IN RAINFALL FROM THE MONTH OF MAY TO THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        sub.append(abs(w1[k] - z1[k]))

    # print(len(flood),len(x1))
    df = pd.DataFrame({'flood': flood})
    df1 = pd.DataFrame({'per_10_days': june})

    x["flood"] = flood
    x["avgjune"] = june
    x["sub"] = sub

    # SAVING THE NEW CSV FILE WITH THE NEW COLOUMNS
    x.to_csv(settings.MEDIA_ROOT + "\\"+"out1.csv")

    df1 = pd.read_csv(settings.MEDIA_ROOT + "\\"+"out1.csv")

    X = df1.iloc[:, [17, 18, 19, 16]].values
    y1 = df1.iloc[:, 20].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y1, test_size=0.3)


    Lr = DecisionTreeClassifier(criterion="entropy")
    Lr.fit(X_train, Y_train)
    pred = Lr.predict(X_test)
    cm = confusion_matrix(pred, Y_test)
    acc = accuracy_score(pred, Y_test)
    fs = f1_score(pred, Y_test)
    score = Lr.score(X, y1)
    return score,fs

def randomforestal():
    # importing the dataset..
    x = pd.read_csv(filepath)

    # checking the null values in the dataset...

    x['JAN'] = x['JAN'].fillna(x['JAN'].mean())
    x['FEB'] = x['FEB'].fillna(x['FEB'].mean())
    x['MAR'] = x['MAR'].fillna(x['MAR'].mean())
    x['APR'] = x['APR'].fillna(x['APR'].mean())
    x['MAY'] = x['MAY'].fillna(x['MAY'].mean())
    x['JUN'] = x['JUN'].fillna(x['JUN'].mean())
    x['JUL'] = x['JUL'].fillna(x['JUL'].mean())
    x['AUG'] = x['AUG'].fillna(x['AUG'].mean())
    x['SEP'] = x['SEP'].fillna(x['SEP'].mean())
    x['OCT'] = x['OCT'].fillna(x['OCT'].mean())
    x['NOV'] = x['NOV'].fillna(x['NOV'].mean())
    x['DEC'] = x['DEC'].fillna(x['DEC'].mean())
    x['ANNUAL'] = x['ANNUAL'].fillna(x['ANNUAL'].mean())
    x['Jan-Feb'] = x['Jan-Feb'].fillna(x['Jan-Feb'].mean())
    x['Mar-May'] = x['Mar-May'].fillna(x['Mar-May'].mean())
    x['Jun-Sep'] = x['Jun-Sep'].fillna(x['Jun-Sep'].mean())
    x['Oct-Dec'] = x['Oct-Dec'].fillna(x['Oct-Dec'].mean())

    y1 = list(x["YEAR"])
    x1 = list(x["Jun-Sep"])
    z1 = list(x["JUN"])
    w1 = list(x["MAY"])

    flood = []
    june = []
    sub = []

    # CREATING A NEW COLOUMN WITH BINARY CLASSIFICATION DEPENDING IF THAT YEAR HAD FLOODED OR NOT, USING RAINFALL OF THAT YEAR AS THRESHOLD
    # print(x1[114])
    for i in range(0, len(x1)):
        if x1[i] > 2400:
            flood.append('1')
        else:
            flood.append('0')

    # print(len(x1))

    # APPROAXIMATELY FINDING THE RAINFALL DATA FOR 10 DAYS FOR THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        june.append(z1[k] / 3)

    # FINDING THE INCREASE IN RAINFALL FROM THE MONTH OF MAY TO THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        sub.append(abs(w1[k] - z1[k]))

    # print(len(flood),len(x1))
    df = pd.DataFrame({'flood': flood})
    df1 = pd.DataFrame({'per_10_days': june})

    x["flood"] = flood
    x["avgjune"] = june
    x["sub"] = sub

    # SAVING THE NEW CSV FILE WITH THE NEW COLOUMNS
    x.to_csv(settings.MEDIA_ROOT + "\\" + "out1.csv")

    df1 = pd.read_csv(settings.MEDIA_ROOT + "\\" + "out1.csv")

    X = df1.iloc[:, [17, 18, 19, 16]].values
    y1 = df1.iloc[:, 20].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y1, test_size=0.3)
    from sklearn.ensemble import RandomForestClassifier
    Lr = RandomForestClassifier(n_estimators=20, criterion="entropy")
    Lr.fit(X_train, Y_train)
    pred = Lr.predict(X_test)
    cm = confusion_matrix(pred, Y_test)
    acc = accuracy_score(pred, Y_test)
    fs = f1_score(pred, Y_test)
    score = Lr.score(X, y1)
    return score, fs

def ann_al():
    # importing the dataset..
    x = pd.read_csv(filepath)

    # checking the null values in the dataset...

    x['JAN'] = x['JAN'].fillna(x['JAN'].mean())
    x['FEB'] = x['FEB'].fillna(x['FEB'].mean())
    x['MAR'] = x['MAR'].fillna(x['MAR'].mean())
    x['APR'] = x['APR'].fillna(x['APR'].mean())
    x['MAY'] = x['MAY'].fillna(x['MAY'].mean())
    x['JUN'] = x['JUN'].fillna(x['JUN'].mean())
    x['JUL'] = x['JUL'].fillna(x['JUL'].mean())
    x['AUG'] = x['AUG'].fillna(x['AUG'].mean())
    x['SEP'] = x['SEP'].fillna(x['SEP'].mean())
    x['OCT'] = x['OCT'].fillna(x['OCT'].mean())
    x['NOV'] = x['NOV'].fillna(x['NOV'].mean())
    x['DEC'] = x['DEC'].fillna(x['DEC'].mean())
    x['ANNUAL'] = x['ANNUAL'].fillna(x['ANNUAL'].mean())
    x['Jan-Feb'] = x['Jan-Feb'].fillna(x['Jan-Feb'].mean())
    x['Mar-May'] = x['Mar-May'].fillna(x['Mar-May'].mean())
    x['Jun-Sep'] = x['Jun-Sep'].fillna(x['Jun-Sep'].mean())
    x['Oct-Dec'] = x['Oct-Dec'].fillna(x['Oct-Dec'].mean())

    y1 = list(x["YEAR"])
    x1 = list(x["Jun-Sep"])
    z1 = list(x["JUN"])
    w1 = list(x["MAY"])

    flood = []
    june = []
    sub = []

    # CREATING A NEW COLOUMN WITH BINARY CLASSIFICATION DEPENDING IF THAT YEAR HAD FLOODED OR NOT, USING RAINFALL OF THAT YEAR AS THRESHOLD
    # print(x1[114])
    for i in range(0, len(x1)):
        if x1[i] > 2400:
            flood.append('1')
        else:
            flood.append('0')

    # print(len(x1))

    # APPROAXIMATELY FINDING THE RAINFALL DATA FOR 10 DAYS FOR THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        june.append(z1[k] / 3)

    # FINDING THE INCREASE IN RAINFALL FROM THE MONTH OF MAY TO THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        sub.append(abs(w1[k] - z1[k]))

    # print(len(flood),len(x1))
    df = pd.DataFrame({'flood': flood})
    df1 = pd.DataFrame({'per_10_days': june})

    x["flood"] = flood
    x["avgjune"] = june
    x["sub"] = sub

    # SAVING THE NEW CSV FILE WITH THE NEW COLOUMNS
    x.to_csv(settings.MEDIA_ROOT + "\\" + "out1.csv")

    df1 = pd.read_csv(settings.MEDIA_ROOT + "\\" + "out1.csv")

    X = df1.iloc[:, [17, 18, 19, 16]].values
    y1 = df1.iloc[:, 20].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y1, test_size=0.3)


    # adding the input output layers and selecting activation functions..
    classifier = Sequential()
    classifier.add(Dense(3, activation="relu", kernel_initializer='uniform', input_dim=4))
    # adding one more hidden layer..
    classifier.add(Dense(2, activation="relu", kernel_initializer='uniform'))
    # adding ouput layer...
    classifier.add(Dense(1, activation="sigmoid", kernel_initializer='uniform'))
    # compiling ANN...
    classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    # fitting data to ANN
    classifier.fit(X_train,Y_train, batch_size=4, epochs=10)
    # prediction..
    y_pred = classifier.predict_classes(X_test)
    fs=f1_score(y_pred,Y_test)
    score=accuracy_score(y_pred,Y_test)

    return score,fs


def prediction_results(rain1,rain2,rain3,rain4):
    # importing the dataset..
    x = pd.read_csv(filepath)
    # checking the null values in the dataset...
    x['JAN'] = x['JAN'].fillna(x['JAN'].mean())
    x['FEB'] = x['FEB'].fillna(x['FEB'].mean())
    x['MAR'] = x['MAR'].fillna(x['MAR'].mean())
    x['APR'] = x['APR'].fillna(x['APR'].mean())
    x['MAY'] = x['MAY'].fillna(x['MAY'].mean())
    x['JUN'] = x['JUN'].fillna(x['JUN'].mean())
    x['JUL'] = x['JUL'].fillna(x['JUL'].mean())
    x['AUG'] = x['AUG'].fillna(x['AUG'].mean())
    x['SEP'] = x['SEP'].fillna(x['SEP'].mean())
    x['OCT'] = x['OCT'].fillna(x['OCT'].mean())
    x['NOV'] = x['NOV'].fillna(x['NOV'].mean())
    x['DEC'] = x['DEC'].fillna(x['DEC'].mean())
    x['ANNUAL'] = x['ANNUAL'].fillna(x['ANNUAL'].mean())
    x['Jan-Feb'] = x['Jan-Feb'].fillna(x['Jan-Feb'].mean())
    x['Mar-May'] = x['Mar-May'].fillna(x['Mar-May'].mean())
    x['Jun-Sep'] = x['Jun-Sep'].fillna(x['Jun-Sep'].mean())
    x['Oct-Dec'] = x['Oct-Dec'].fillna(x['Oct-Dec'].mean())
    y1 = list(x["YEAR"])
    x1 = list(x["Jun-Sep"])
    z1 = list(x["JUN"])
    w1 = list(x["MAY"])
    flood = []
    june = []
    sub = []

    # CREATING A NEW COLOUMN WITH BINARY CLASSIFICATION DEPENDING IF THAT YEAR HAD FLOODED OR NOT, USING RAINFALL OF THAT YEAR AS THRESHOLD
    # print(x1[114])
    for i in range(0, len(x1)):
        if x1[i] > 2400:
            flood.append('1')
        else:
            flood.append('0')

    # print(len(x1))

    # APPROAXIMATELY FINDING THE RAINFALL DATA FOR 10 DAYS FOR THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        june.append(z1[k] / 3)

    # FINDING THE INCREASE IN RAINFALL FROM THE MONTH OF MAY TO THE MONTH OF JUNE IN EVERY YEAR FROM 1901 TO 2015
    for k in range(0, len(x1)):
        sub.append(abs(w1[k] - z1[k]))

    # print(len(flood),len(x1))
    df = pd.DataFrame({'flood': flood})
    df1 = pd.DataFrame({'per_10_days': june})

    x["flood"] = flood
    x["avgjune"] = june
    x["sub"] = sub

    # SAVING THE NEW CSV FILE WITH THE NEW COLOUMNS
    x.to_csv(settings.MEDIA_ROOT + "\\" + "out1.csv")

    df1 = pd.read_csv(settings.MEDIA_ROOT + "\\" + "out1.csv")

    X = df1.iloc[:, [17, 18, 19, 16]].values
    y1 = df1.iloc[:, 20].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, y1, test_size=0.3)

    Lr = LogisticRegression()

    Lr.fit(X_train, Y_train)
    #arr = np.array(i)
    re= Lr.predict([[rain1,rain2,rain3,rain4]])
    return re