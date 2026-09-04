# Production Scenario Walkthrough: Real-Time Diamond Valuation Engine

This document walks through a real-world production scenario illustrating why modern Machine Learning teams rely on **Feast** instead of ad-hoc SQL queries and custom pipelines.

---

## 1. The Business Context & Architecture

### Company: *CaratValuate* (Online Luxury Diamond Exchange)
*CaratValuate* is an online marketplace where diamond dealers list inventory and buyers purchase diamonds in real-time.

To maintain market trust and maximize liquidity, the platform requires an **Automated Diamond Price Appraisal Service**:
* When a user browses diamonds or an API client queries prices by ID, the system must return an instant price prediction with **sub-50ms latency**.
* The model must be retrained weekly on historical transaction data without **data leakage** or **feature drift**.

---

## 2. The Problems Without a Feature Store

Before introducing Feast, the engineering team faced three major architectural bottlenecks:

```mermaid
flowchart TD
    subgraph Without Feast
        A1["Team 1: Physical Lab DB"] -->|Custom SQL| DS1["Data Scientist 1: Training Script"]
        A2["Team 2: Gemology Lab DB"] -->|Ad-hoc Pandas Merge| DS1
        A3["Team 3: Transactions DB"] -->|Manual Joins (Data Leakage Risk)| DS1
        
        DS1 -->|Trained Model| PROD["Production API"]
        
        A1 -->|Live DB Queries (Slow 2-3s Latency)| PROD
        A2 -->|Live DB Queries| PROD
        PROD -->|Training-Serving Skew| ERR["Price Prediction Errors & Timeouts"]
    end
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

```mermaid
flowchart TD
    subgraph Upstream Ingestion
        P["Laser Lab (physical_features.parquet)"]
        Q["Gemology Lab (quality_features.parquet)"]
        L["Sales DB (labels.parquet)"]
    end

    subgraph Feast Feature Store
        REG["Central Metadata Registry (registry.db)"]
        DEF["Feature Definitions as Code (definitions.py)"]
        DEF -->|feast apply| REG
    end

    subgraph Offline Path: Training (Point-in-Time Correct)
        L -->|Time-Travel Joins| FEAST_HIST["store.get_historical_features()"]
        P --> FEAST_HIST
        Q --> FEAST_HIST
        FEAST_HIST --> TRAIN_DATA["training_set.parquet"]
        TRAIN_DATA --> TRAIN_SCRIPT["train_model.py"]
        TRAIN_SCRIPT --> MODEL["model.joblib"]
    end

    subgraph Online Path: Low-Latency Inference
        P -->|feast materialize| HOT_CACHE["SQLite / Redis Online Store"]
        Q -->|feast materialize| HOT_CACHE
        
        CLIENT["User / API Request: diamond_id = 101"] --> API["FastAPI / Inference Service"]
        HOT_CACHE -->|5ms Key-Value Lookup| API
        MODEL --> API
        API --> RESP["Predicted Price: $2,945"]
    end
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

## 5. Deep Dive: Why Do We Send IDs? (How Feature Lookup Works)

A common question is: **Why does the client send only `diamond_id` (e.g., `101, 202`) instead of the full feature values? How does this capture the latest trends and changes?**

### 1. Separation of Concerns (Thin Clients)
In real-world applications (web apps, mobile apps, or external APIs), the user's browser or frontend service only knows *which* diamond is being viewed (e.g., `diamond_id: 101`). 
The client:
- Does not know raw physical dimensions (`x`, `y`, `z`, `table`, `depth`).
- Does not have direct database access to gemological certification records.
- Should not pass 20+ feature columns over HTTP, which would increase payload size, latency, and security risks.

### 2. Fetching the Most Recent State (Handling Fluctuations & Trend Updates)
Features for an entity are not static:
- A diamond might be **re-cut or polished** (updating its physical measurements `carat`, `table`, `x`, `y`, `z`).
- A diamond might receive a **re-certified optical grade** (updating `cut`, `color`, `clarity`).
- In extended setups, entities often have **rolling trend features** (e.g., `avg_market_price_7d`, `inventory_days_on_market`, `demand_surge_index`).

**How Feast Handles This:**
1. Upstream batch or stream jobs update the source tables whenever changes occur.
2. `feast materialize` runs periodically (e.g., hourly, daily, or via streaming).
3. The online store overwrites the record for `diamond_id: 101` with its **freshest, most up-to-date snapshot**.
4. When `store.get_online_features(entity_rows=[{"diamond_id": 101}])` is called, Feast instantly returns the **most recent feature values and latest trends**, without recalculating anything on the fly.

### 3. Summary: ID-Based Retrieval Workflow
```text
Frontend Client (Sends only diamond_id: 101)
       │
       ▼
Inference Service
       │
       ▼ (Key-Value lookup by diamond_id)
Feast Online Store (SQLite / Redis)
       │──> Pulls the latest materialized snapshot of all 9 features
       ▼
Model Execution: model.predict(latest_features)
       │
       ▼
Instant Response: $2,945 (returned in <10ms)
```

---

## 6. Summary of Business Impact

| Metric / Dimension | Before Feast (Ad-hoc Joins) | With Feast Feature Store |
| :--- | :--- | :--- |
| **Online Serving Latency** | 2,000ms – 3,000ms (Heavy SQL joins) | **< 10ms** (Key-Value Online Store) |
| **Data Leakage Risk** | High (accidental future joins) | **Zero** (Point-in-time time-travel joins) |
| **Training-Serving Skew** | Frequent (different offline/online logic) | **Eliminated** (Single feature definition layer) |
| **Engineering Overhead** | Redundant SQL scripts across teams | **Centralized Registry & UI** |
