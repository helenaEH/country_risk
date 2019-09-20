"""
Using random forest to determine indicators that are relevant for predicting crisis in a country 1 year in advance
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import shap
# Load the data into a dataframe
df = pd.read_csv('data.csv')

# drop NANs
df.dropna(subset=['PCRISIS1'], inplace=True)

# Extract the dependend and independent variables from the dataframe
y = df['PCRISIS1']
X = df.drop(labels=['ID', 'COUNTRY', 'PCRISIS1'], axis=1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit a Random Forest Classifier
m = RandomForestClassifier(n_estimators=500, random_state=42)
m.fit(X_train, y_train)

# Evaluate the model
m_train = m.score(X_train, y_train)
m_cv = cross_val_score(m, X_train, y_train, cv=5).mean()
m_test = m.score(X_test, y_test)
print('Evaluation of the Random Forest performance\n')
print(f'Training score: {m_train.round(4)}')
print(f'Cross validation score: {m_cv.round(4)}')
print(f'Test score: {m_test.round(4)}')

# Create a SHAP explainer
explainer = shap.TreeExplainer(m)
shap_values = explainer.shap_values(X)

# Plot feature importance
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig('feature_importance.jpg')
plt.show()
