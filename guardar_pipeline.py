import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1) Cargar CSV (TU RUTA REAL)
df = pd.read_csv(r"C:\Users\marin\Downloads\Smartphone_Usage_Productivity_Dataset_50000.csv")

# 2) Quitar ID
df = df.drop(columns=["User_ID"])

target = "Work_Productivity_Score"
X = df.drop(columns=[target])
y = df[target]

cat_cols = ["Gender", "Occupation", "Device_Type"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

pipe.fit(X, y)

# 3) Guardar EN EL ESCRITORIO
joblib.dump(pipe, r"C:\Users\marin\OneDrive\Escritorio\pipeline.joblib")

print("✅ OK: pipeline.joblib guardado en el Escritorio")
