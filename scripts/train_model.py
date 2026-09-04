import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_parquet("feature_repo/data/training_set.parquet")

feature_cols = ["carat", "depth", "table", "x", "y", "z",
                "cut", "color", "clarity"]
X, y = df[feature_cols], df["price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200, max_depth=15, n_jobs=-1, random_state=42
)
model.fit(X_tr, y_tr)

mae = mean_absolute_error(y_te, model.predict(X_te))
print(f"Test MAE: ${mae:.0f}")

joblib.dump(model, "feature_repo/data/model.joblib")
print("Saved model -> feature_repo/data/model.joblib")
