# 1. Import Library
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 2. Load Dataset
df = pd.read_csv("titanic.csv") 

print(df.head())
print(df.info())

# 3. EDA Singkat
print("\nJumlah missing value:")
print(df.isnull().sum())

print("\nStatistik deskriptif:")
print(df.describe())

# 4. Preprocessing
df.drop('name', axis=1, inplace=True)

df = pd.get_dummies(df, columns=['sex'], drop_first=True)

print("\nDataset setelah preprocessing:")
print(df.head())

# 5. Split Data
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Build Model Decision Tree
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Evaluasi Model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Visualisasi Pohon
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Survived", "Survived"],
    filled=True
)
plt.show()
