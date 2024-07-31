from sklearn.ensemble import ExtraTreesClassifier as xt

from sklearn.metrics import accuracy_score as score

import sklearn.datasets as dt

from sklearn.model_selection import train_test_split as tts

model = xt()

X, y = dt.make_circles(n_samples=1000)

xtr, xts, ytr, yts = tts(X, y)

model.fit(xtr, ytr)

print(score(yts, model.predict(xts)))
