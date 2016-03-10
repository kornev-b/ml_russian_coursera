from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV as grid_search
from scipy.sparse import csr_matrix

ng = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space'],
                    download_if_missing=True)
tfidf_vr = TfidfVectorizer()
# print "tfidfing..."
tfidf = tfidf_vr.fit_transform(raw_documents=ng.data, y=ng.target)
tfidf_vr.tran
# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# cv = KFold(len(ng.data), n_folds=5, shuffle=True, random_state=241)
# clf = SVC(kernel='linear', random_state=241)
# gs = grid_search(clf, grid, scoring='accuracy', cv=cv)
# print "fitting..."
# gs.fit(X=tfidf, y=ng.target)
# for a in gs.grid_scores_:
#     print a.mean_validation_score
#     print a.parameters
#     print "\n"
clf = SVC(C=1.0, kernel='linear', random_state=241)
clf.fit(X=tfidf, y=ng.target)
abs_coef = abs(clf.coef_)
sorted_coef = np.argsort(abs_coef.toarray()[0])[:-11:-1]
fn = tfidf_vr.get_feature_names()
answer = []
for i in sorted_coef:
    answer.append(fn[i])
for x in sorted(answer):
    print x