from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
from statistics import mean, median
import numpy as np
import sys
import pseudoExtractor as ps
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
classifier =svm.SVC(kernel='poly')
KNeighborsClassifier(n_neighbors=3)#(n_neighbors=3)default#n_neighbors=5
#svm.SVC()#svm.SVC(gamma='auto')
#RandomForestClassifier(n_estimators=30, max_depth=10, random_state=0)

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


##################apply different ML algorithms###############################


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

clf = classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)


# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_auc_score # for printing AUC


print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
 
print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test.round(),y_pred)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)



