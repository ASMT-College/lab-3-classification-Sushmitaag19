import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('db.csv')

df = df.drop(columns=['Unnamed: 9'], errors='ignore')
df = df.dropna(subset=['Outcome'])

X = df.drop(columns='Outcome').fillna(df.mean(numeric_only=True))
y = df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))