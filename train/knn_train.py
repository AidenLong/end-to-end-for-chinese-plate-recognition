# -*- coding utf-8 -*- #

import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# data = pd.read_excel("train_data.xlsx")
# with open('train_data.pkl', 'wb') as file:
#     pickle.dump(data, file)

with open('train_data.pkl', 'rb') as file:
    data = pickle.load(file)

x = data.iloc[:, 2:3].join(data.iloc[:, 58:])
y = data.iloc[:, 3:58]

# one-hot处理
age = pd.get_dummies(data["age"])
age_keys = age.keys()
for key in age_keys:
    x[key] = age[key]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.head())
print(y_train.head())

# classifier = MLkNN(k=20)
# classifier.fit(x_train, y_train)
# with open('ml-knn.model', 'wb') as file:
#     pickle.dump(classifier, file)

classifier = OneVsRestClassifier(SVC(kernel='linear'))
classifier.fit(x_train, y_train)
with open('svm.model', 'wb') as file:
    pickle.dump(classifier, file)

# predict
predictions = classifier.predict(x_test)

print(accuracy_score(y_test, predictions))
print("准确率:", classifier.score(x_test, predictions))
