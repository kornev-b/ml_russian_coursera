#!/usr/bin/env python

import numpy
import pandas

from datetime import datetime

from scipy.sparse import hstack

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

def measure(func):
	utcnow = datetime.utcnow()
	x = func()
	return x, datetime.utcnow() - utcnow

def lr_with_C_search(X, y, validator, random_state):
	gs = GridSearchCV(
		LogisticRegression(random_state=random_state),
		{ 'C': numpy.power(10.0, numpy.arange(-5, 6)) },
		scoring='roc_auc',
		cv=KFold(len(y), 5, True))

	gs_results = measure(lambda: gs.fit(X, y).best_params_['C'])
	print '\tBest C = %f, time = %s' % gs_results

	lr = LogisticRegression(C = gs_results[0], random_state=random_state)
	lr_results = measure(lambda: cross_val_score(lr, X, y, 'roc_auc', validator).mean())
	print '\tQuality: %.2f, time: %s' % lr_results
	print
	return (lr_results[0], lr)

features = pandas.read_csv('./features.csv', index_col='match_id')
random_state = None
gradient_boosting_verbose = False

match_results_columns = [
	'duration',
	'radiant_win',
	'tower_status_radiant',
	'tower_status_dire',
	'barracks_status_radiant',
	'barracks_status_dire']
X = features.drop(match_results_columns, axis=1)
y = features.radiant_win

matches_count = len(features)
data_counts = X.count()
data_with_nulls = data_counts[data_counts < matches_count]

print 'Features with missing data:'
print
for k,v in data_with_nulls.iteritems():
	print '\t%s = %2.2f%%' % (k, (matches_count - v) * 100.0 / matches_count)
print

print 'Target feature: radiant_win'
print

X = X.fillna(0)

models = list()
validator = KFold(len(X), 5, True, random_state)
estimation_results = dict()
estimator_counts = [10, 20, 30]

print 'GradientBoostingClassifier'
print

for estimator_count in estimator_counts:
	clf = GradientBoostingClassifier(n_estimators=estimator_count, verbose=gradient_boosting_verbose, random_state=random_state)
	if estimator_count not in estimation_results:
		estimation_results[estimator_count] = measure(lambda: cross_val_score(clf, X, y, 'roc_auc', validator).mean())
		models.append((estimation_results[estimator_count][0], clf))

for k,v in sorted(estimation_results.iteritems()):
	print '\tTrees: %d, quality: %.2f, time: %s' % (k, v[0], str(v[1]))
print

print 'LogisticRegression, boosting-scheme'
models.append(lr_with_C_search(X, y, validator, random_state))

print 'LogisticRegression, dropping categorical features'

categorical_features = [
	'lobby_type',
	'r1_hero',
	'r2_hero',
	'r3_hero',
	'r4_hero',
	'r5_hero',
	'd1_hero',
	'd2_hero',
	'd3_hero',
	'd4_hero',
	'd5_hero'
]
X_no_categories = X.drop(categorical_features, axis=1)
models.append(lr_with_C_search(X_no_categories, y, validator, random_state))

print 'LogisticRegression, bag of heroes'

hero_features = [
	'r1_hero',
	'r2_hero',
	'r3_hero',
	'r4_hero',
	'r5_hero',
	'd1_hero',
	'd2_hero',
	'd3_hero',
	'd4_hero',
	'd5_hero'
]

bag = set()
for player in hero_features:
	bag = bag.union(features[player].unique())

X_pick = numpy.zeros((features.shape[0], max(max(bag), len(bag))))

for i, match_id in enumerate(features.index):
	for p in xrange(5):
		X_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
		X_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_bag_heroes = hstack([X_no_categories, X_pick])
models.append(lr_with_C_search(X_bag_heroes, y, validator, random_state))

if random_state is None:
	print 'Result models when random_state is none'
else:
	print 'Result models when random_state is %d' % random_state
print
for model in sorted(models):
	print 'Quality: %.2f, model: %s' % model

best_model = max(models)
test_data = pandas.read_csv('features_test.csv', index_col='match_id')
test_data = test_data.fillna(0)

# best model is Gradient, so we do not need any additional transforms
X = pandas.read_csv('./features.csv', index_col='match_id')
X = X.drop(match_results_columns, axis=1).fillna(0)
best_model[1].fit(X, y)
y_test = best_model[1].predict_proba(test_data)[:, 1]
test_data['radiant_win'] = y_test
test_data.radiant_win.to_csv('prediction.csv', header=True)