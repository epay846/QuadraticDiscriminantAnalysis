import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.__check_buildimport matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
train_voweldf = pd.read_csv('vowels.csv')
test_voweldf = pd.read_csv('voweldf_test.csv')

X_train = train_voweldf.drop(columns=train_voweldf.columns[1]).values
y_train = train_voweldf.iloc[:, 1].values
X_test = test_voweldf.drop(columns=test_voweldf.columns[1]).values
y_test = test_voweldf.iloc[:, 1].values

qdamoel = QuadraticDiscriminantAnalysis()
qdamodel.fit(X_train, y_train)
y_pred = qdamodel.predict(X_test)
misclassification_error = 1 - accuracy_score(y_test, y_pred)
print(misclassification_error)

cm = confusion_matriix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heamap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
