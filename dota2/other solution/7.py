import pandas as pn
import numpy as np
import sklearn as skl
import time
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#read train features
features_train = pn.read_csv('features.csv', index_col='match_id')

#deleting excess features
features_train.drop('duration', axis=1, inplace=True)
features_train.drop('tower_status_radiant', axis=1, inplace=True)
features_train.drop('tower_status_dire', axis=1, inplace=True)
features_train.drop('barracks_status_radiant', axis=1, inplace=True)
features_train.drop('barracks_status_dire', axis=1, inplace=True)

target_train = features_train['radiant_win']
features_train.drop('radiant_win', axis=1, inplace=True)


#detecting skips
print("1.1")
print("============")
featuresLen = len(features_train)

index = 0
for row in features_train.count():
    if(row != featuresLen):
        print(features_train.columns[index] + " " + str(row))
    index = index + 1
print("============")
print("")

#fillna
features_train.fillna(0, inplace=True)

stCa = skl.preprocessing.StandardScaler()
stCa.fit(features_train, target_train)
features_trainSt = stCa.transform(features_train)

#target
print("1.2")
print("============")
print("radiant_win")
print("============")
print("")

print("1.3")
print("============")
kFold = skl.cross_validation.KFold(featuresLen, n_folds = 5, shuffle = True)
for n in [1,2,3,5,8,10,15,20,30, 35, 40]:
    start_time = datetime.datetime.now()
    cls = GradientBoostingClassifier(n_estimators=n)
    score = skl.cross_validation.cross_val_score(cls, features_trainSt, target_train, cv = kFold, scoring='roc_auc').mean()
    print(datetime.datetime.now() - start_time)
    print(str(n) + ": " + str(score))
print("============")
print("")



features_train = pn.read_csv('features.csv', index_col='match_id')
features_train.drop('duration', axis=1, inplace=True)
features_train.drop('tower_status_radiant', axis=1, inplace=True)
features_train.drop('tower_status_dire', axis=1, inplace=True)
features_train.drop('barracks_status_radiant', axis=1, inplace=True)
features_train.drop('barracks_status_dire', axis=1, inplace=True)

target_train = features_train['radiant_win']
features_train.drop('radiant_win', axis=1, inplace=True)




featuresLen = len(features_train)
features_train.fillna(0, inplace=True)

stCa = skl.preprocessing.StandardScaler()
stCa.fit(features_train, target_train)
features_trainSt = stCa.transform(features_train)

print("2.1")
print("============")
kFold = skl.cross_validation.KFold(featuresLen, n_folds = 5, shuffle = True)
for c in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
    start_time = datetime.datetime.now()
    cls = LogisticRegression(penalty='l2', C=c)
    score = skl.cross_validation.cross_val_score(cls, features_trainSt, target_train, cv = kFold, scoring='roc_auc').mean()
    print(datetime.datetime.now() - start_time)
    print(str(c) + ": " + str(score))
print("============")
print("")



print("2.2")
print("============")

featuresWC_train = features_train.drop('lobby_type', axis=1, inplace=False)
featuresWC_train.drop('r1_hero', axis=1, inplace=True)
featuresWC_train.drop('r2_hero', axis=1, inplace=True)
featuresWC_train.drop('r3_hero', axis=1, inplace=True)
featuresWC_train.drop('r4_hero', axis=1, inplace=True)
featuresWC_train.drop('r5_hero', axis=1, inplace=True)
featuresWC_train.drop('d1_hero', axis=1, inplace=True)
featuresWC_train.drop('d2_hero', axis=1, inplace=True)
featuresWC_train.drop('d3_hero', axis=1, inplace=True)
featuresWC_train.drop('d4_hero', axis=1, inplace=True)
featuresWC_train.drop('d5_hero', axis=1, inplace=True)

stCa = skl.preprocessing.StandardScaler()
stCa.fit(featuresWC_train, target_train)
featuresWCSt_train = stCa.transform(featuresWC_train)


kFold = skl.cross_validation.KFold(featuresLen, n_folds = 5, shuffle = True)
for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 1000, 100000]:
    start_time = datetime.datetime.now()
    cls = LogisticRegression(penalty='l2', C=c)
    score = skl.cross_validation.cross_val_score(cls, featuresWCSt_train, target_train, cv = kFold, scoring='roc_auc').mean()
    print(datetime.datetime.now() - start_time)
    print(str(c) + ": " + str(score))
print("============")
print("")

print("2.3")
print("============")
features_trainSource = pn.read_csv('features.csv', index_col='match_id')
AllHeroes = features_trainSource['r1_hero']
for seria in ['r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']:
    AllHeroes.add(features_trainSource[seria])
uniqIn = AllHeroes.unique()
countUn = len(AllHeroes.unique())
print(countUn)
print("============")
print("")

print("2.4-2.5")
print("============")
X_pick = np.zeros((features_trainSource.shape[0], countUn))

for i, match_id in enumerate(features_trainSource.index):
    for p in xrange(5):
        X_pick[i, uniqIn.tolist().index(features_trainSource.ix[match_id, 'r%d_hero' % (p+1)])-1] = 1
        X_pick[i, uniqIn.tolist().index(features_trainSource.ix[match_id, 'd%d_hero' % (p+1)])-1] = -1



featuresWCStNew_train = np.concatenate([featuresWCSt_train, X_pick], axis=1)
kFold = skl.cross_validation.KFold(featuresLen, n_folds = 5, shuffle = True)
for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 1000, 100000]:
    start_time = datetime.datetime.now()
    cls = LogisticRegression(penalty='l2', C=c)
    score = skl.cross_validation.cross_val_score(cls, featuresWCStNew_train, target_train, cv = kFold, scoring='roc_auc').mean()
    print(datetime.datetime.now() - start_time)
    print(str(c) + ": " + str(score))

print("============")
print("")


print("2.6")
print("============")
features_test = pn.read_csv('features_test.csv', index_col='match_id')

X_pick = np.zeros((features_test.shape[0], countUn))

for i, match_id in enumerate(features_test.index):
    for p in xrange(5):
        X_pick[i, uniqIn.tolist().index(features_test.ix[match_id, 'r%d_hero' % (p+1)])-1] = 1
        X_pick[i, uniqIn.tolist().index(features_test.ix[match_id, 'd%d_hero' % (p+1)])-1] = -1


features_test.drop('lobby_type', axis=1, inplace=True)
features_test.drop('r1_hero', axis=1, inplace=True)
features_test.drop('r2_hero', axis=1, inplace=True)
features_test.drop('r3_hero', axis=1, inplace=True)
features_test.drop('r4_hero', axis=1, inplace=True)
features_test.drop('r5_hero', axis=1, inplace=True)
features_test.drop('d1_hero', axis=1, inplace=True)
features_test.drop('d2_hero', axis=1, inplace=True)
features_test.drop('d3_hero', axis=1, inplace=True)
features_test.drop('d4_hero', axis=1, inplace=True)
features_test.drop('d5_hero', axis=1, inplace=True)

features_test.fillna(0, inplace=True)

stCa = skl.preprocessing.StandardScaler()
stCa.fit(features_test, target_train)
features_test = stCa.transform(features_test)

features_test_New = np.concatenate([features_test, X_pick], axis=1)

start_time = datetime.datetime.now()

cls = LogisticRegression(penalty='l2', C=0.1)
cls.fit(featuresWCStNew_train, target_train)
predicts = cls.predict_proba(features_test_New)
max = -1
min = 2
for pred in predicts:
    if(max)< pred[0]:
        max = pred[0]
    if(max)< pred[1]:
        max = pred[1]
    if(min)> pred[1]:
        min = pred[1]
    if(min)> pred[0]:
        min = pred[0]
    if(min<0):
        print("Warning. <0")
    if(max>1):
        print("Warning. >0")
print("max: "+str(max)+" min: "+str(min))
print(datetime.datetime.now() - start_time)
print("============")
print("")

