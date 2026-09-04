# 1. Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run dataset preparation (writes to feature_repo/data)
python3 scripts/prepare_diamond_features.py

# 3. Apply feature definitions
feast -c feature_repo apply

# 4. Push feature values into the online store (materialize)
feast -c feature_repo materialize 2024-01-01T00:00:00 $(date -u +"%Y-%m-%dT%H:%M:%S")

# 5. Build training dataset from offline store
python3 scripts/build_training_set.py

# 6. Train model
python3 scripts/train_model.py

# 7. Test online inference / feature retrieval
python3 scripts/serve_online.py

# Optional: Reset everything to scratch for next demo
# rm -rf feature_repo/data && mkdir -p feature_repo/data
