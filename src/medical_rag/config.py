"""Configuration loading for the medical RAG experiment.

Will load and merge YAML configuration from config/default.yaml,
config/models.yaml, and config/conditions.yaml into typed config objects
consumed by the rest of the pipeline.

TODO: define pydantic models for experiment, model, and condition config.
TODO: load_config(path) -> merges a config file with defaults.
TODO: load_models_config() -> returns available model/endpoint definitions.
TODO: load_conditions_config() -> returns experiment condition definitions.
"""
