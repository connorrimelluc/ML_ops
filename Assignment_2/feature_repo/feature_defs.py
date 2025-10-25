from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, UnixTimestamp

v1_source = FileSource(path="../data_work/v1.parquet", timestamp_field="event_ts")
v2_source = FileSource(path="../data_work/v2.parquet", timestamp_field="event_ts")

athlete = Entity(name="row_id", join_keys=["row_id"])

athletes_v1 = FeatureView(
    name="athletes_v1",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="feat_num_1", dtype=Float32),
        Field(name="feat_num_2", dtype=Float32),
        Field(name="event_ts", dtype=UnixTimestamp),
        Field(name="row_id", dtype=Int64),
    ],
    source=v1_source,
)

athletes_v2 = FeatureView(
    name="athletes_v2",
    entities=[athlete],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="feat_num_1", dtype=Float32),
        Field(name="feat_num_2", dtype=Float32),
        Field(name="feat_num_1_x_2", dtype=Float32),
        Field(name="feat_num_1_sq", dtype=Float32),
        Field(name="event_ts", dtype=UnixTimestamp),
        Field(name="row_id", dtype=Int64),
    ],
    source=v2_source,
)
