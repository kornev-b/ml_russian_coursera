import pandas as pd
from sklearn.linear_model import Perceptron as perc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler as scale

df_train = pd.read_csv('train.csv', index_col=None)
df_test = pd.read_csv('test.csv', index_col=None)

y_train = df_train[df_train.columns[0]]
x_train = df_train.drop(df_train.columns[0], axis=1)
x_test = df_test.drop(df_test.columns[0], axis=1)
y_test = df_test[df_test.columns[0]]
pc = perc(random_state=241)
clf = pc.fit(X=x_train, y=y_train)
predictions = clf.predict(x_test)
accuracy_unscaled = accuracy_score(y_true=y_test, y_pred=predictions)
print "unscaled = " + accuracy_unscaled.__str__()
scaler = scale()
x_train_scaled = scaler.fit_transform(X=x_train)
x_test_scaled = scaler.transform(X=x_test)
clf = pc.fit(X=x_train_scaled, y=y_train)
predictions = clf.predict(x_test_scaled)
accuracy_scaled = accuracy_score(y_true=y_test, y_pred=predictions)
print "scaled = " + accuracy_scaled.__str__()
print 1.0 * (accuracy_scaled - accuracy_unscaled)
