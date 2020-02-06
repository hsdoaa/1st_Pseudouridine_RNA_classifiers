from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
from statistics import mean, median
import numpy as np
import sys
import pseudoExtractor as ps
from sklearn.model_selection import train_test_split

sys.path.insert(1, "./ml/")
import ml.svm as svm

#start pseudoExtractor 
controlHela, pseudoHela = ps.get_Hela()   #?????  this function returns two panadas data frame???
#omit file name
drp = [0, 2]
controlHela = controlHela.drop(drp, axis=1) #drop 1st and second column
pseudoHela = pseudoHela.drop(drp, axis=1)

print(controlHela.iloc[0,1])

kmerData = []
#!cut data to 500
#for i in range(len(controlHela)):
for i in range(len(controlHela)):
    kmer = controlHela.iloc[i, 0] # T select first column and assign it to kemer https://stackoverflow.com/questions/31593201/how-are-iloc-ix-and-loc-different
    kmerData.append([kmer])

    values = controlHela.iloc[i, 1] #to select 2nd column and assign it to values 

    sig = ""
    for j in range(len(values)):
        if values[j] == '_':
            #convert to int
            kmerData[i].append(int(sig))
            sig = ""

        elif j == (len(values) - 1):
            sig += values[j]
            kmerData[i].append(int(sig))
            sig = ""

        else:
            sig += values[j]
        


pseudoKmerData = []
for i in range(len(pseudoHela)):
    kmer = pseudoHela.iloc[i, 0]
    pseudoKmerData.append([kmer])

    values = pseudoHela.iloc[i, 1]
    sig = ""
    for j in range(len(values)):
        if values[j] == '_':
            #convert to int
            pseudoKmerData[i].append(int(sig))
            sig = ""

        elif j == (len(values) - 1):
            sig += values[j]
            pseudoKmerData[i].append(int(sig))
            sig = ""

        else:
            sig += values[j]

X = []
Xval = []
Y = []
Yval = []

#get random indexes
#total = len(controlHela) + len(pseudoHela)
totalControl = 300
prevIndexes = np.random.choice(len(controlHela), 360, replace=False)
#set length to 300(random choices)
kmerData = np.array(kmerData)[prevIndexes]
#kmerData = kmerData[:360]
print("size of ", len(kmerData))
total = 360 + len(pseudoHela)
indexes = np.random.choice(total, total, replace=False)


for i in range(len(kmerData)):
    X.append(kmerData[i][0])


for i in range(len(pseudoKmerData)):
    X.append(pseudoKmerData[i][0]) # X store kmerData and pseudoKmerData


le = preprocessing.LabelEncoder()
le.fit(X)
print(le.classes_)
X = le.transform(X)
X = X.reshape(-1, 1)

#onehot encode
enc = OneHotEncoder(handle_unknown='ignore', n_values=350)
enc.fit(X)
onehots = enc.transform(X).toarray()
X = onehots


for i in range(len(kmerData)):
    Xval.append([mean(kmerData[i][1:]), median(kmerData[i][1:]), max(kmerData[i][1:]), min(kmerData[i][1:])])
    Yval.append([0])


for i in range(len(pseudoKmerData)):
    Xval.append([mean(pseudoKmerData[i][1:]), median(pseudoKmerData[i][1:]), max(pseudoKmerData[i][1:]), min(pseudoKmerData[i][1:])])
    Yval.append([1])


#insert one hot Feature
for i in range(len(Xval)):
    for j in range(len(X[i])):
        Xval[i].append(X[i][j])


#randomize indexes
X = np.array(Xval)[indexes]
Y = np.array(Yval)[indexes]
print(len(X), len(Y))

print("**************",X.shape)
print("**************",Y.shape)

####################################################define the Keras model##############################################

# Evaluate the model: Model Accuracy, how often is the classifier correct
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import SGD
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_auc_score # for printing AUC


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

model = Sequential()
model.add(Dense(12, input_dim=354, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.05, momentum=0.99), metrics=['accuracy'])

# Fit the model                            #epochs= ﬁxed number of iterations through the dataset called epochs
model.fit(X_train, y_train, validation_split=0.2, epochs=150, batch_size=16) #batch_size=the number of instances that are evaluated before a weight update


# evaluate the keras model on the same dataset, but the dataset can be devided into training and testing, then fit the model on the training and evalaute the model on testing
_, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = model.predict_classes(X_test)



y_pred = model.predict_classes(X_test) 
print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test.round(),y_pred)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)



