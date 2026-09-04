import joblib
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
model = joblib.load("feature_repo/data/model.joblib")

# in prod these ids arrive from an HTTP request
incoming_ids = [101, 202, 303, 404, 5050]

# fetch the latest features from the ONLINE store
online_features = store.get_online_features(
    features=[
        "diamond_physical:carat",
        "diamond_physical:depth",
        "diamond_physical:table",
        "diamond_physical:x",
        "diamond_physical:y",
        "diamond_physical:z",
        "diamond_quality:cut",
        "diamond_quality:color",
        "diamond_quality:clarity",
    ],
    entity_rows=[{"diamond_id": i} for i in incoming_ids],
).to_dict()

df = pd.DataFrame(online_features)
feature_cols = ["carat", "depth", "table", "x", "y", "z",
                "cut", "color", "clarity"]
preds = model.predict(df[feature_cols])

print(f"{'ID':>6}  {'carat':>6}  {'cut':>4}  {'color':>5}  {'predicted':>12}")
print("-" * 45)
for diamond_id, carat, cut, color, pred in zip(
    incoming_ids, df["carat"], df["cut"], df["color"], preds
):
    print(f"{diamond_id:>6}  {carat:>6.2f}  {cut:>4}  {color:>5}  ${pred:>10,.0f}")
