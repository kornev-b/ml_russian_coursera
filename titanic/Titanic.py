import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# def func(x):
#     if x.find('(') != -1:
#         return x[x.index('(') + 1:]
#     else:
#         return x


data = pandas.read_csv('titanic.csv', index_col=None)
# female = data.loc[data['Sex'] == 'female']
# names = pandas.DataFrame()
# names['name'], names['call'] = zip(*female['Name'].apply(lambda x: x.split(', ', 1)))
# names['prefix'], names['husband?'] = zip(*names['call'].apply(lambda x: x.split('. ')))
# print names['husband?'].apply(lambda x: func(x)).apply(lambda x: x.split(' ')[0]).value_counts()

df = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
df = df.dropna()
df['SexInt'] = df['Sex'].map({'female':0, 'male':1})
del df['Sex']
df = df.reset_index(drop=True)
dts = DecisionTreeClassifier(random_state=241)
dts.fit(df[['Pclass', 'Fare', 'Age', 'SexInt']], df['Survived'])
print dts.feature_importances_
