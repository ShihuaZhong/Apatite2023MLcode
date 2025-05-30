from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

X = np.loadtxt("C:/Users/qw123/Desktop/PAPER/IS/data.data/dataset.X.txt", dtype=np.float64)
y = np.loadtxt("C:/Users/qw123/Desktop/PAPER/IS/data.data/dataset.Y.txt", dtype=np.float64)
X = np.log10(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=48, shuffle=True, stratify=y)

scaler = StandardScaler()
scaler.fit(x_train)
xt = scaler.transform(x_train)
xte = scaler.transform(x_test)
#
from collections import Counter
from imblearn.over_sampling import SMOTE
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=40)
X_res, y_res = sm.fit_resample(xt, y_train)
print('Resampled dataset shape %s' % Counter(y_res))

svm = SVC(C=2.0, kernel="linear", gamma=0.5, class_weight='balanced')
svm.fit(X_res, y_res)



logreg = SVC()
score = cross_val_score(logreg,X_res, y_res,cv=5)
print(score)
print (score.mean())

y_pred = svm.predict(xte)
score = svm.score(xte, y_test)
分类 = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


x1= np.loadtxt("C:/Users/qw123/Desktop/PAPER/IS/data.data/predX.txt", dtype=np.float64)
y1 = np.loadtxt("C:/Users/qw123/Desktop/PAPER/IS/data.data/predY.txt", dtype=np.float64)
x2 = np.log10(x1)

x2 = scaler.transform(x2)
y_pred1 = svm.predict(x2)
独立验证 = confusion_matrix(y1, y_pred1)
print(classification_report(y1, y_pred1))

score1  = accuracy_score(y1, y_pred1)
print("Accuracy:", score1)