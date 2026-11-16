"""Run the MLB, NFL and CFB prediction pipeline on the homelab instead of as Cloud Functions.

Standard library only: these modules run under the pipeline's own venv with PYTHONPATH pointing at /opt/mllab/src, and import nothing
heavy themselves. The pipeline code (pandas, xgboost, BigQuery) runs in a subprocess against a staged bundle. Plan: docs/HOMELAB_PIPELINE.md."""
