from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X = np.loadtxt("dataset.X.txt", dtype=np.float64)
y = np.loadtxt("dataset.Y.txt", dtype=np.float64)
X = np.log10(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=48, shuffle=True, stratify=y)

scaler = StandardScaler()
scaler.fit(x_train)
xt = scaler.transform(x_train)
xte = scaler.transform(x_test)

svm = SVC(C=1.0, kernel="linear", gamma=0.0009765625, class_weight='balanced')
svm.fit(xt, y_train)

logreg = SVC()
score = cross_val_score(logreg,xt,y_train,cv=5)
print(score)
print (score.mean())

y_pred = svm.predict(xte)
score = svm.score(xte, y_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))


x1 = np.loadtxt("predX.txt", dtype=np.float64)
y1 = np.loadtxt("predY.txt", dtype=np.float64)
x2 = np.log10(x1)

x2 = scaler.transform(x2)
y_pred1 = svm.predict(x2)
cm1 = confusion_matrix(y1, y_pred1)
print(classification_report(y1, y_pred1))

