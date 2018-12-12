# Schema for saving data and putting into Cassandra

TARGET_SCHEMA = {
    "table_name": "target_categories",
    "options": {
        "primary_key": ["model_id"],
    }
}