import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

df = pd.read_csv('classification.csv', index_col=None)
tp = 0
fp = 0
tn = 0
fn = 0
y_true = df[df.columns[0]]
y_pred = df[df.columns[1]]
for i in xrange(len(df.index)):
    expected = y_true[i]
    predicted = y_pred[i]
    if (expected & predicted) == 1: tp += 1
    elif (expected | predicted) != 1: tn += 1
    elif (expected ^ predicted) == 1:
        fp += predicted
        fn += expected
print "tp=" + tp.__str__()
print "fp=" + fp.__str__()
print "tn=" + tn.__str__()
print "fn=" + fn.__str__()

print "accuracy=" + accuracy_score(y_true=y_true, y_pred=y_pred).__str__()
print "precision=" + precision_score(y_true=y_true, y_pred=y_pred).__str__()
print "recall=" + recall_score(y_true=y_true, y_pred=y_pred).__str__()
print "f1-score=" + f1_score(y_true=y_true, y_pred=y_pred).__str__()

print "next..."
df = pd.read_csv('scores.csv', index_col=None)
y_true = df[df.columns[0]]
score_logreg = df[df.columns[1]]
score_svm = df[df.columns[2]]
score_knn = df[df.columns[3]]
score_tree = df[df.columns[4]]
print "score_logreg: roc_auc_score=" + roc_auc_score(y_true=y_true, y_score=score_logreg).__str__()
print "score_svm: roc_auc_score=" + roc_auc_score(y_true=y_true, y_score=score_svm).__str__()
print "score_knn: roc_auc_score=" + roc_auc_score(y_true=y_true, y_score=score_knn).__str__()
print "score_tree: roc_auc_score=" + roc_auc_score(y_true=y_true, y_score=score_tree).__str__()
print "score_logreg: precision_recall_curve:"
precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=score_logreg)
dic = {'precision': precision, 'recall': recall}
df = pd.DataFrame.from_dict(data=dic)
df = df[df.recall >= 0.7].groupby('precision')
print df
print "score_svm: precision_recall_curve:"
precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=score_svm)
dic = {'precision': precision, 'recall': recall}
df = pd.DataFrame.from_dict(data=dic)
df = df[df.recall >= 0.7]
df.sort_values(by='precision')
print df
print "score_knn: precision_recall_curve:"
precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=score_knn)
dic = {'precision': precision, 'recall': recall}
df = pd.DataFrame.from_dict(data=dic)
df = df[df.recall >= 0.7]
df.sort_values(by='precision')
print df
print "score_tree: precision_recall_curve:"
precision, recall, thresholds = precision_recall_curve(y_true=y_true, probas_pred=score_tree)
dic = {'precision': precision, 'recall': recall}
df = pd.DataFrame.from_dict(data=dic)
df = df[df.recall >= 0.7]
df.sort_values(by='precision')
print df
