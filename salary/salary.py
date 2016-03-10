from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.linear_model import Ridge

df_train = pd.read_csv('salary-train.csv', index_col=None)
df_test = pd.read_csv('salary-train.csv', index_col=None)
df_test['FullDescription'] = df_test['FullDescription'].lower()
print df_test
df_train['LocationNormalized'].fillna('nan', inplace=True)
df_train['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
x_train_categ = enc.fit_transform(df_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
x_test_categ = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))