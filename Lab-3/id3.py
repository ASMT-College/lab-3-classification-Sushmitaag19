import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('db.csv')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(subset=['Outcome'])

X = df.drop(columns='Outcome').fillna(df.mean(numeric_only=True))
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))