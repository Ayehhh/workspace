import joblib
import pandas as pd

# Load model pipeline
model = joblib.load(r"C:\Users\U10096633\OneDrive - BASF\Desktop\MDC - local\PBi\test iris\model.joblib")

# 1. Tentukan senarai nama lajur
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# 2. Ambil data dan convert ke numeric
X_new = dataset[feature_cols].apply(pd.to_numeric, errors='coerce')

# 3. Buat prediction guna 'X_new.values' (sebab nama variable kat atas ialah X_new)
dataset['prediction'] = model.predict(X_new.values)
