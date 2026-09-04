from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# diamonds
# each row is a diamond

# 1. define that row
# describe features
diamond = Entity(
    name="diamond",
    join_keys=["diamond_id"],
    value_type=ValueType.INT64,
    description="A single diamond identified by its row id",
)

physical_source = FileSource(
    name="physical_source",
    path="data/physical_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

quality_source = FileSource(
    name="quality_source",
    path="data/quality_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# exists, applied to this entity
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

quality_fv = FeatureView(
    name="diamond_quality",
    entities=[diamond],
    ttl=timedelta(days=400),
    schema=[
        Field(name="cut",     dtype=Int64),
        Field(name="color",   dtype=Int64),
        Field(name="clarity", dtype=Int64),
    ],
    source=quality_source,
    online=True,
)
