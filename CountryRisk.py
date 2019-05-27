# Country Risk essay Random Forest project
# What indicators are the most relevant for predicting crises in a country, 1 year in advance

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import pydot
from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()

data_all = pd.read_csv(r'C:\Users\helen\PycharmProjects\CountryRisk\data.csv')

# Step 1: deleting rows with missing values
data = data_all.dropna(subset = ['PCRISIS1']) #deletes rows where PCRISIS1 == NaN
print(list(data))
print(data.head())

# Step 2: subsetting the data to only have the independent and target variables included
xVar = list(data.iloc[:,4:17])
yVar = data.iloc[:,3]
data2 = data[xVar]
# transforming continuous variables

# Step 3: splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data2, yVar, test_size=0.2)

# Step 4: building a random forest classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf = clf.fit(X_train, y_train.astype('int'))
print(clf)

# Step 5: measuring the accuracy of predictions on the test data
preds = clf.predict(X_test)
confusion_matrix = pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result'])
print(confusion_matrix)
# confusion matrix accuracy formula = (TP+TN)/(TP+FP+FN+TN)
print(accuracy_score(y_test, preds))

# Step 6: check the feature importance
feature_importance = list(zip(X_train, clf.feature_importances_))
print(feature_importance)

# Step 7: plotting the graph with descending feature importance
names = ['CREDITGDP', 'CAGDP', 'HFCEGDP', 'NPLGROSSLOANS', 'GFGFG', 'ROAA', 'DEPRATE', 'HFCEG', 'PER', 'GDPG', 'INFL', 'REALRATE', 'GFCFGDP']
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(data2.shape[1]), importances[indices])
plt.xticks(range(data2.shape[1]), names, rotation=20, fontsize = 8)
plt.title('Feature Importances')
plt.show()

# Task End

