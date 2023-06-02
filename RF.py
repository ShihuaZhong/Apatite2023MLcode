from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score


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
sm = SMOTE(random_state=48)
X_res, y_res = sm.fit_resample(xt, y_train)
print('Resampled dataset shape %s' % Counter(y_res))

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=41,class_weight='balanced',random_state=48,max_features = 2,max_depth=6)
classifier.fit(X_res, y_res)

logreg = RandomForestClassifier()
score = cross_val_score(logreg,X_res, y_res,cv=10)
print(score)
print (score.mean())

y_pred = classifier.predict(xte)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:", )
print(result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:", result2)

分类 = confusion_matrix(y_test, y_pred)

x1= np.loadtxt("C:/Users/qw123/Desktop/PAPER/IS/data.data/predX.txt", dtype=np.float64)
y1 = np.loadtxt("C:/Users/qw123/Desktop/PAPER/IS/data.data/predY.txt", dtype=np.float64)
x2 = np.log10(x1)

x2 = scaler.transform(x2)
y_pred1 = classifier.predict(x2)
独立验证 = confusion_matrix(y1, y_pred1)
print(classification_report(y1, y_pred1))

result3 = accuracy_score(y1, y_pred1)
print("Accuracy:", result3)