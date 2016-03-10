from sklearn.svm import SVC
import pandas as pd

df = pd.read_csv('svm_data.csv', index_col=None, header=None)
c = SVC(C=100000, kernel='linear', random_state=241)
c.fit(X=df.drop(df.columns[0], axis=1), y=df[df.columns[0]])
print c.support_ + 1