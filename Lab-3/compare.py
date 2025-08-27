import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv('db.csv')

# Drop extra unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(subset=['Outcome'])

X = df.drop(columns='Outcome').fillna(df.mean(numeric_only=True))
y = df['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Decision Tree
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Confusion matrices
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# ROC AUC (better with probabilities)
roc_auc_nb = roc_auc_score(y_test, nb_classifier.predict_proba(X_test)[:, 1])
roc_auc_dt = roc_auc_score(y_test, dt_classifier.predict_proba(X_test)[:, 1])

# Print results
print("\nNaive Bayes vs Decision Tree Classifier Performance:\n")
print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")
print(f"Naive Bayes ROC AUC: {roc_auc_nb:.2f}")
print(f"Decision Tree ROC AUC: {roc_auc_dt:.2f}")
print("\nConfusion Matrix - Naive Bayes:\n", conf_matrix_nb)
print("\nConfusion Matrix - Decision Tree:\n", conf_matrix_dt)
