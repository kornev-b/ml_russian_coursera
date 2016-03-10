import pandas as pd
import sklearn.cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.preprocessing import scale

df = pd.read_csv('wine.data', index_col=None)
target = df[df.columns[0]]
x = df.drop(df.columns[0], axis=1)
kf = cv.KFold(len(df.index), n_folds=5, shuffle=True, random_state=42)
unscaled_result = list()
for i in xrange(1, 50):
    cs_result = cv.cross_val_score(knc(n_neighbors=i), X=x, y=target, cv=kf)
    unscaled_result.append(cs_result.mean())
max_unscaled = max(unscaled_result)
print unscaled_result.index(max_unscaled) + 1
print max_unscaled
scaled_x = scale(X=x)
scaled_result = list()
for i in xrange(1, 50):
    cs_result = cv.cross_val_score(knc(n_neighbors=i), X=scaled_x, y=target, cv=kf)
    scaled_result.append(cs_result.mean())
max_scaled = max(scaled_result)
print scaled_result.index(max_scaled) + 1
print max_scaled

