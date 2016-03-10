from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn import datasets as ds
from sklearn.preprocessing import scale
from numpy import linspace
import sklearn.cross_validation as cv

boston = ds.load_boston()
scaled = scale(boston.data)
kf = cv.KFold(len(scaled), n_folds=5, shuffle=True, random_state=42)
n = linspace(start=1, stop=10, num=200)
result = list()
for i in xrange(1, len(n) - 1):
    knr = KNR(n_neighbors=4, weights='distance', metric='minkowski', p=i)
    cs_result = cv.cross_val_score(knr, X=scaled, y=boston.target, cv=kf, scoring='mean_squared_error')
    result.append(cs_result.mean())
result_max = max(result)
print result.index(result_max) + 1
print result_max