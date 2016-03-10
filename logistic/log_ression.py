import pandas as pd
import math
import numpy
from sklearn.metrics import roc_auc_score

def log_regr(data, w, C):
    sum = 0.0
    for i in xrange(len(data.index)):
        y = data.ix[i, 0]
        x1 = data.ix[i, 1]
        x2 = data.ix[i, 2]
        sum += math.log(1 + math.exp(-y * (w[0] * x1 + w[1] * x2)))
    return 1.0 * sum / len(data.index) + C * numpy.linalg.norm(w) / 2

def gradient(w, wIndex, data, xIndex, C):
    sum = 0.0
    for i in xrange(len(data.index)):
        y = data.ix[i, 0]
        x1 = data.ix[i, 1]
        x2 = data.ix[i, 2]
        sum += y * data.ix[i, xIndex] * (1 - 1. / (1+ math.exp(-y * (w[0] * x1 + w[1] * x2))))
    return w[wIndex] + 1.0 * sum / len(data.index) * 0.1 - 0.1 * C * w[wIndex]

def stop(w, w1):
    return numpy.linalg.norm(w-w1) <= 0.00001

def prob_est(w, data):
    p = []
    for i in xrange(len(data.index)):
        x1 = data.ix[i, 1]
        x2 = data.ix[i, 2]
        a = -w[0] * x1 - w[1] * x2
        a = 1 + math.exp(a)
        p.append(1.0/(a))
    return p

df = pd.read_csv('data-logistic.csv', index_col=None, header=None)
print "Without regularization:"
# w = numpy.array([0.0, 0.0])
# for i in xrange(10000):
#     w1 = numpy.array([0.0, 0.0])
#     w1[0] = gradient(w=w, wIndex=0, data=df, xIndex=1, C=0)
#     w1[1] = gradient(w=w, wIndex=1, data=df, xIndex=2, C=0)
#     print log_regr(data=df, w=w, C=0)
#     if stop(w=w, w1=w1):
#         w = w1
#         print w1
#         break
#     w = w1
# print roc_auc_score(df[df.columns[0]], prob_est(w=w, data=df))
print "With regularization:"
w = numpy.array([0.0, 0.0])
for j in xrange(0, 10000):
    w1 = numpy.array([0.0, 0.0])
    w1[0] = gradient(w=w, wIndex=0, data=df, xIndex=1, C=10)
    w1[1] = gradient(w=w, wIndex=1, data=df, xIndex=2, C=10)
    print log_regr(data=df, w=w, C=10)
    if stop(w=w, w1=w1):
        w = w1
        print w1
        break
    w = w1
print roc_auc_score(df[df.columns[0]], prob_est(w=w, data=df))