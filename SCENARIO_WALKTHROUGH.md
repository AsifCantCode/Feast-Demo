# Production Scenario Walkthrough: Real-Time Diamond Valuation Engine

This document walks through a real-world production scenario illustrating why modern Machine Learning teams rely on **Feast** instead of ad-hoc SQL queries and custom pipelines.

---

## 1. The Business Context & Architecture

### Company: *CaratValuate* (Online Luxury Diamond Exchange)
*CaratValuate* is an online marketplace where diamond dealers list inventory and buyers purchase diamonds in real-time.

To maintain market trust and maximize liquidity, the platform requires an **Automated Diamond Price Appraisal Service**:
* When a user browses diamonds or an API client queries prices by ID, the system must return an instant price prediction with **sub-50ms latency**.
* The model must be retrained periodically on historical transaction data without **data leakage** or **feature drift**.

---

## 2. The Problems Without a Feature Store

Before introducing Feast, the engineering team faced three major architectural bottlenecks:

```text
[ Team 1: Physical DB ] ──(Custom SQL)──────────┐
[ Team 2: Gemology DB ] ──(Ad-hoc Pandas Merge)─┼──> [ Data Scientist: Model Training ]
[ Team 3: Sales DB    ] ──(Accidental Leakage)──┘               │
                                                                ▼
                                                       [ Production API ]
                                                                ▲
[ Team 1: Physical DB ] ──(Slow 2-3s SQL Queries on Click)──────┤
[ Team 2: Gemology DB ] ──(Slow 2-3s SQL Queries on Click)──────┘
                                                                │
                                                                ▼
                                              [ Timeouts & Prediction Errors ]
```

### Problem 1: Data Silos & Duplicate Engineering
- **Team 1 (Intake & Laser Scanning):** Measures physical geometry (`carat`, `depth`, `table`, `x`, `y`, `z`) upon warehouse arrival.
- **Team 2 (Gemological Certification):** Evaluates optical clarity, cut, and color (`cut`, `color`, `clarity`) days later after laboratory testing.
- **Team 3 (Transactions):** Records historical sale prices and auction bids.

Without a feature store, every data scientist re-wrote complex SQL queries and custom encodings to join these disparate databases, leading to duplicate code and inconsistencies.

### Problem 2: Data Leakage in Historical Training
Diamonds are re-evaluated over time (e.g., re-graded after recutting or updated lab inspection). 
A simple SQL join `ON diamond_id` joins a sale that occurred in **May** with a quality grade updated in **June** (future data). The model appears accurate in offline evaluation but underperforms in production.

### Problem 3: Production Latency & Training-Serving Skew
During live inference, querying transactional SQL databases across multiple tables took 2 to 3 seconds. Additionally, feature transformation code written in pandas for training often differed slightly from production inference code, causing silent model degradation (**training-serving skew**).

---

## 3. The Feast Solution: End-to-End Practical Workflow

With Feast, features are treated as **version-controlled software assets**. The same feature definitions power both point-in-time historical training (offline) and sub-millisecond real-time serving (online).

```text
                    UPSTREAM DATA INGESTION
    [ Laser Lab ]         [ Gemology Lab ]         [ Sales DB ]
 (physical.parquet)       (quality.parquet)      (labels.parquet)
         │                       │                      │
         └───────────────┬───────┴──────────────────────┘
                         │
                         ▼
             [ FEAST FEATURE STORE ]
             - Registry (registry.db)
             - Definitions (definitions.py)
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼ (Offline Path)                ▼ (Online Path: feast materialize)
 [ Point-in-Time Join ]          [ Low-Latency Online Store ]
 (get_historical_features)       (SQLite / Redis Hot Cache)
         │                               │
         ▼                               │
[ training_set.parquet ]                 │ (5ms Key-Value Lookup)
         │                               │
         ▼                               ▼
 [ train_model.py ] ──> [ model.joblib ] ──> [ FastAPI Inference Service ]
                                                      ▲
                                                      │ (Request: diamond_id = 101)
                                              [ User / Client App ]
```

---

## 4. Step-by-Step Scenario Execution

Here is how the scenario plays out using the files in this repository:

### Phase 1: Upstream Teams Emit Data
Each team publishes their partitioned dataset asynchronously.
```bash
python3 scripts/prepare_diamond_features.py
```
* **Result:** `feature_repo/data/physical_features.parquet`, `quality_features.parquet`, and `labels.parquet` are generated with realistic `event_timestamp` metadata.

---

### Phase 2: Centralizing Feature Definitions
The ML Platform team defines entities and feature views declaratively in `feature_repo/definitions.py`:
```python
diamond = Entity(name="diamond", join_keys=["diamond_id"])

physical_fv = FeatureView(
    name="diamond_physical",
    entities=[diamond],
    ttl=timedelta(days=400),
    schema=[
        Field(name="carat", dtype=Float32),
        Field(name="depth", dtype=Float32),
        Field(name="table", dtype=Float32),
        Field(name="x",     dtype=Float32),
        Field(name="y",     dtype=Float32),
        Field(name="z",     dtype=Float32),
    ],
    source=physical_source,
    online=True,
)
```
The definitions are applied to the infrastructure:
```bash
feast -c feature_repo apply
```

---

### Phase 3: Building Point-in-Time Training Sets (Zero Leakage)
When building training data, Feast matches every historical label event with the exact feature state valid at that point in time:
```bash
python3 scripts/build_training_set.py
```
* **Why it matters:** Even if a diamond's quality was updated later, Feast only pulls the features known *before* the sale happened, eliminating future data leakage.

---

### Phase 4: Training and Model Serialization
The model trains on the leakage-free historical dataset:
```bash
python3 scripts/train_model.py
```
* **Result:** Outputs a serialized model (`feature_repo/data/model.joblib`) with a validated Test MAE (~$267).

---

### Phase 5: Materializing to the Online Store
To enable real-time lookup, batch feature data is loaded into the high-performance online store:
```bash
feast -c feature_repo materialize 2024-01-01T00:00:00 $(date -u +"%Y-%m-%dT%H:%M:%S")
```
* **Result:** SQLite (or Redis in cloud production) is populated with the latest feature values indexed by `diamond_id`.

---

### Phase 6: Real-Time Serving in Production
When a customer views diamonds on the frontend, the service sends only the IDs (`[101, 202, 303]`):
```bash
python3 scripts/serve_online.py
```
* **Code in Action:**
  ```python
  online_features = store.get_online_features(
      features=[
          "diamond_physical:carat", "diamond_physical:depth", "diamond_physical:table",
          "diamond_physical:x", "diamond_physical:y", "diamond_physical:z",
          "diamond_quality:cut", "diamond_quality:color", "diamond_quality:clarity",
      ],
      entity_rows=[{"diamond_id": 101}, {"diamond_id": 202}],
  ).to_dict()
  ```
* **Result:** The online store returns pre-joined, pre-processed features in **<5ms**, and the model outputs predictions immediately:
  ```text
      ID   carat   cut  color     predicted
  ---------------------------------------------
     101    0.75     3      5  $     2,945
     202    0.70     3      5  $     2,664
     303    0.78     4      6  $     2,976
  ```

---

## 5. Frequently Asked Questions (Core Concepts Clarified)

### Q1: "Why do we need to train a model if Feast already stores features? Why can't we just fetch the price?"
* **Features vs. Target (Price):**
  Feast stores **input features** (`carat`, `cut`, `clarity`, `depth`, `table`, etc.). Feast does **NOT** store the price of newly listed, unsold diamonds.
* **Why an ML Model is Necessary:**
  When a dealer brings 500 brand new diamonds to the marketplace, **nobody knows what fair price they will sell for**. 
  If the price was already fixed in a database, you would just do `SELECT price FROM diamonds WHERE id = 101` and would not need Machine Learning. 
  The ML model uses the physical and quality features retrieved by Feast to **estimate / predict** fair market valuation ($ price).

---

### Q2: "How do we input new data into the system, and how does Feast handle it?"
There are two types of new data entering a production ML system:

#### 1. New Diamonds to be Appraised (Inference Pipeline)
When 1,000 new diamonds arrive at the warehouse:
1. Upstream scanners write their physical and quality features into the data source files (`physical_features.parquet` and `quality_features.parquet`).
2. Run `feast materialize` (or Feast streaming/push sources in production).
3. The new diamond IDs (e.g., `diamond_id: 60001`) and their feature snapshots are immediately inserted into the Online Store (SQLite/Redis).
4. Clients can now query prices for these new diamonds instantly via `store.get_online_features(entity_rows=[{"diamond_id": 60001}])`.

#### 2. New Historical Sales Transactions (Training Pipeline)
As diamonds are actually bought and sold over time:
1. New transaction records (e.g., `diamond_id`, `actual_sold_price`, `sale_timestamp`) are appended to `labels.parquet`.
2. To adapt to macroeconomic changes (inflation, seasonal diamond demand spikes), the team periodically runs:
   - `python3 scripts/build_training_set.py` (performs time-travel joins on the new historical data)
   - `python3 scripts/train_model.py` (retrains and saves the new `model.joblib`)

---

### Q3: "Why do we send only IDs during online inference? Does it fetch the latest trends?"
When the client calls `store.get_online_features(entity_rows=[{"diamond_id": 101}])`, sending only the ID provides two advantages:

1. **Thin Clients & Security:** The client device (browser/mobile app) only knows *which* diamond is on screen (`diamond_id: 101`). It does not need to download or send 20 internal lab columns over the network.
2. **Latest Snapshot & Trend Freshness:** If a diamond was re-polished yesterday (changing `table` width) or re-graded in the lab (changing `clarity`), `feast materialize` has already updated that record in the online store. Feast always retrieves the **most recent valid feature state** for that ID in single-digit milliseconds.

```text
Frontend Client (Sends only diamond_id: 101)
       │
       ▼
Inference Service
       │
       ▼ (Key-Value lookup by diamond_id in <5ms)
Feast Online Store (SQLite / Redis)
       │──> Returns freshest snapshot: [carat=0.75, cut=3, color=5, depth=61.5...]
       ▼
Model Execution: model.predict(latest_features)
       │
       ▼
Instant Response: $2,945
```

---

## 6. Summary of Business Impact

| Metric / Dimension | Before Feast (Ad-hoc Joins) | With Feast Feature Store |
| :--- | :--- | :--- |
| **Online Serving Latency** | 2,000ms – 3,000ms (Heavy SQL joins) | **< 10ms** (Key-Value Online Store) |
| **Data Leakage Risk** | High (accidental future joins) | **Zero** (Point-in-time time-travel joins) |
| **Training-Serving Skew** | Frequent (different offline/online logic) | **Eliminated** (Single feature definition layer) |
| **Engineering Overhead** | Redundant SQL scripts across teams | **Centralized Registry & UI** |
