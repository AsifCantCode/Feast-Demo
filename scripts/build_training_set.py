import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

labels = pd.read_parquet("feature_repo/data/labels.parquet")

training_df = store.get_historical_features(
    entity_df=labels,
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
).to_df()

print(training_df.head())
print(f"\nShape: {training_df.shape}")
training_df.to_parquet("feature_repo/data/training_set.parquet", index=False)
print("Saved -> feature_repo/data/training_set.parquet")
