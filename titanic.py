import pandas as pd
import seaborn as sns

# Load Titanic dataset directly from seaborn
titanic = sns.load_dataset('titanic')
print(titanic.head())

# Check basic info
print(titanic.info())

# Summary statistics
print(titanic.describe())

# Count missing values
print(titanic.isnull().sum())

# Drop columns with too many missing values
titanic = titanic.drop(columns=['deck'])

# Fill missing age values with median
titanic['age'] = titanic['age'].fillna(titanic['age'].median())

# Fill missing embark_town with mode
titanic['embark_town'] = titanic['embark_town'].fillna(titanic['embark_town'].mode()[0])

# Survival by gender
print(titanic.groupby('sex')['survived'].mean())

# Survival by class
print(titanic.groupby('class')['survived'].mean())

import matplotlib.pyplot as plt

# Survival by gender
sns.barplot(x='sex', y='survived', data=titanic)
plt.title("Survival Rate by Gender")
plt.show()

# Survival by class
sns.barplot(x='class', y='survived', data=titanic)
plt.title("Survival Rate by Class")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Select features
features = ['sex', 'age', 'class']
titanic_model = titanic.copy()

# Convert categorical variables to numeric
titanic_model = pd.get_dummies(titanic_model[features], drop_first=True)

X = titanic_model
y = titanic['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
