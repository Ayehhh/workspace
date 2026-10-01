import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import accuracy_score

# Power BI dah sediakan data dalam pembolehubah 'dataset'
# iris = pd.read_csv("Iris.csv")  <-- Tak perlu baris ni lagi

X = dataset.drop('species', axis=1)
y = dataset['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test Model
y_pred = model.predict(X_test)

# 3. 
accuracy = accuracy_score(y_test, y_pred)

# Save model ke disk
import os

output_directory = r"C:\Users\U10096633\OneDrive - BASF\Desktop\MDC - local\PBi\test iris"

# Save file ke folder tersebut
full_path = os.path.join(output_directory, "model.joblib")
joblib.dump(model, full_path)

print(f"Model berjaya disave di: {full_path}")
joblib.dump(model, "model.joblib")

# Power BI perlukan output dalam bentuk DataFrame untuk dipaparkan
output_df = dataset.copy()
output_df['predicted_species'] = model.predict(X)
